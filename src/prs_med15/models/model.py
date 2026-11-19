"""PRS-Med component orchestration with turn-ordered entity memory."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ..training.rollout import (
    HistoryUsage,
    choose_dialogue_history,
    choose_history_mask,
)
from .encoder import FeatureBundle, _patch_valid
from .llm_bridge import SlotSpan, build_inputs_embeds, forward_hidden, forward_last_hidden, lm_head_logits
from .region_memory import EntityEntry, EntityRegistry


@dataclass
class PRSMedOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor
    mask_logits: list[torch.Tensor]
    actions: list[dict[str, Any]]
    labels: torch.Tensor
    history_fractions: dict[str, float]
    earlier_memory_detached: bool = True


class PRSMed15(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        adapter: nn.Module,
        global_pooler: nn.Module,
        query_encoder: nn.Module,
        detail_retriever: nn.Module,
        language_model: nn.Module,
        mask_decoder: nn.Module,
        region_pool: nn.Module,
        region_tokenizer: nn.Module,
        slot_token_ids: dict[str, int],
        history_mode: str = "oracle",
        oracle_fraction: float = 0.5,
        history_generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter
        self.global_pooler = global_pooler
        self.query_encoder = query_encoder
        self.detail_retriever = detail_retriever
        self.language_model = language_model
        self.mask_decoder = mask_decoder
        self.region_pool = region_pool
        self.region_tokenizer = region_tokenizer
        self.slot_token_ids = slot_token_ids
        self.history_mode = history_mode
        self.oracle_fraction = oracle_fraction
        self.history_generator = history_generator
        # False when the text loss weight is 0 (stage A): the per-turn full-prefix LLM pass only
        # produces CE logits, so it is skipped (mask path and memory are unaffected).
        self.compute_text_logits = True
        # One LLM pass per turn (through the end of the assistant answer) serves both the [SEG]
        # states and the CE logits. The turn's own memory tokens come after the answer, carry no
        # labels and cannot influence earlier positions, so this equals the old two-pass scheme.
        self.fused_turn_pass = True
        self._region_cache: dict[tuple[str, str, int], torch.Tensor] | None = None
        if history_mode not in {"oracle", "predicted", "mixed"}:
            raise ValueError("invalid history mode")

    @staticmethod
    def _depth_positions(
        geometry: list[Any],
        depth: int,
        device: torch.device,
    ) -> torch.Tensor:
        values = []
        for geo in geometry:
            if (
                isinstance(geo, dict)
                and not geo.get("unknown_geometry", False)
                and "spacing_mm" in geo
            ):
                start = geo.get("slab_start", 0)
                spacing = float(geo["spacing_mm"][0])
                values.append((torch.arange(depth, device=device) + start) * spacing)
            else:
                values.append(torch.arange(depth, device=device))
        return torch.stack(values)

    @staticmethod
    def _reference_parts(reference: str, default_scan: str) -> tuple[str, str]:
        if ":" in reference:
            scan_id, entity_id = reference.split(":", 1)
            return scan_id, entity_id
        return default_scan, reference

    def _region_rows(self, registry: EntityRegistry, scan_id: str, entity_id: str) -> torch.Tensor:
        entry = registry.get(scan_id, entity_id)
        cache_key = (scan_id, entity_id, int(entry.version))
        if self._region_cache is not None and cache_key in self._region_cache:
            return self._region_cache[cache_key]
        rows = self.region_tokenizer(entry)
        expected = self.region_tokenizer.tokens_per_entity
        if rows.ndim != 2 or rows.shape[0] != expected:
            raise ValueError(
                f"entity {scan_id}:{entity_id} produced {rows.shape[0]} region rows; "
                f"expected {expected}"
            )
        if self._region_cache is not None:
            self._region_cache[cache_key] = rows
        return rows

    def _build_prefix(
        self,
        batch: dict[str, Any],
        batch_index: int,
        stop: int,
        turn_index: int,
        dense: torch.Tensor,
        dense_valid: torch.Tensor,
        global_tokens: torch.Tensor,
        registry: EntityRegistry,
        detail_cache: dict[int, torch.Tensor],
    ):
        item = batch["items"][batch_index]
        ids = batch["input_ids"][batch_index : batch_index + 1, :stop]
        labels = batch["labels"][batch_index : batch_index + 1, :stop]
        attention = batch["attention_mask"][batch_index : batch_index + 1, :stop]
        continuous: dict[str, torch.Tensor] = {}
        spans: list[SlotSpan] = []

        global_positions = torch.where(ids[0] == self.slot_token_ids["global"])[0]
        if global_positions.numel():
            if len(global_positions) != len(global_tokens):
                raise ValueError(
                    f"global slot mismatch: {len(global_positions)} positions, "
                    f"{len(global_tokens)} rows"
                )
            key = f"global_{batch_index}"
            continuous[key] = global_tokens
            spans.append(SlotSpan(0, global_positions, key))

        for current_turn in range(turn_index + 1):
            start, turn_stop = item["turn_boundaries"][current_turn]
            positions = torch.where(
                ids[0, start : min(turn_stop, stop)] == self.slot_token_ids["detail"]
            )[0] + start
            if not positions.numel():
                continue
            values = detail_cache[current_turn]
            if len(positions) != len(values):
                raise ValueError(
                    f"turn {current_turn} detail slot mismatch: "
                    f"{len(positions)} positions, {len(values)} rows"
                )
            key = f"detail_{batch_index}_{current_turn}"
            continuous[key] = values
            spans.append(SlotSpan(0, positions, key))

        for record_index, record in enumerate(item.get("memory_records", [])):
            positions = [position for position in record["positions"] if position < stop]
            if not positions:
                continue
            expected = self.region_tokenizer.tokens_per_entity
            if len(positions) != expected:
                raise ValueError(
                    f"memory record for {record['scan_id']}:{record['entity_id']} has "
                    f"{len(positions)} in-prefix slots; expected {expected}"
                )
            rows = self._region_rows(
                registry,
                record["scan_id"],
                record["entity_id"],
            )
            key = f"region_{batch_index}_{record_index}"
            continuous[key] = rows
            spans.append(SlotSpan(0, positions, key))

        return build_inputs_embeds(
            ids,
            spans,
            continuous,
            self.language_model.get_input_embeddings(),
            attention,
            labels,
        )

    def _detail_for_turn(
        self,
        item: dict[str, Any],
        turn_index: int,
        registry: EntityRegistry,
        dense: torch.Tensor,
        dense_valid: torch.Tensor,
        device: torch.device,
        prepared=None,
    ) -> torch.Tensor:
        question_ids = item["question_ids"][turn_index].to(device).unsqueeze(0)
        if not question_ids.numel():
            question_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
        region_entries = []
        references = item.get("reference_entity_ids", item.get("references", []))
        for reference in references[turn_index]:
            scan_id, entity_id = self._reference_parts(reference, "S1")
            # Only entities that already exist (created in EARLIER turns, with a mask) may condition
            # this turn's retrieval; same-turn or not-in-view entities are skipped (no leakage).
            if not registry.contains(scan_id, entity_id):
                continue
            region_entries.append(self._region_rows(registry, scan_id, entity_id))
        region_tensor = None
        if region_entries:
            region_tensor = torch.cat(region_entries, dim=0).unsqueeze(0)
        queries = self.query_encoder(
            question_ids,
            region_entries=region_tensor,
        )
        if prepared is None:
            return self.detail_retriever(queries, dense, dense_valid)[0]
        return self.detail_retriever.attend(queries, prepared)[0]

    def _prior_mask(
        self,
        action: dict[str, Any],
        registry: EntityRegistry,
    ) -> torch.Tensor | None:
        references = action.get("references", [])
        if not references:
            return None
        scan_id, entity_id = self._reference_parts(references[0], action["scan_id"])
        return registry.get(scan_id, entity_id).mask

    @staticmethod
    def _text_stop(labels: torch.Tensor, turn_start: int, turn_stop: int) -> int:
        """End of the supervised assistant text in a turn (memory tokens follow it unlabelled)."""
        supervised = torch.nonzero(labels[turn_start:turn_stop] != -100)
        if not supervised.numel():
            return turn_stop
        return turn_start + int(supervised[-1]) + 1

    def _decode_actions(self, hidden, current_actions, batch_index, registry, selection, dense,
                        bundle, valid, images, geometry, usage, action_outputs, mask_outputs) -> None:
        for action_index, action in current_actions:
            seg_state = hidden[0, action["seg_token_pos"]]
            prior_mask = self._prior_mask(action, registry)
            logits = self.mask_decoder(
                dense[batch_index].permute(0, 3, 1, 2),
                seg_state,
                images.shape[-2:],
                valid[batch_index],
                prior_mask=prior_mask,
            )
            predicted = torch.sigmoid(logits[:, 0]) > 0.5
            chosen_mask, provenance = choose_history_mask(
                action.get("target"),
                predicted,
                selection,
            )
            pooled, confidence = self.region_pool(
                dense[batch_index],
                chosen_mask.to(dense.device),
                bundle.valid_mask[batch_index],
            )
            if action["op"] == "update":
                old = registry.get(action["scan_id"], action["entity_id"])
                version = old.version + 1
            else:
                version = 1
            entry = EntityEntry(
                action["scan_id"],
                action["entity_id"],
                version,
                action["turn_id"],
                seg_state.detach(),
                chosen_mask.detach(),
                pooled.detach(),
                geometry[batch_index],
                confidence,
                provenance,
            )
            if action["op"] == "update":
                registry.update(action["scan_id"], action["entity_id"], entry)
            else:
                registry.create(entry)
            usage.add(provenance, chosen_mask, action.get("target"))
            action_outputs[action_index] = {
                **action,
                "logits": logits,
                "valid": valid[batch_index],
            }
            mask_outputs[action_index] = logits

    def forward(self, batch: dict[str, Any]) -> PRSMedOutput:
        images = batch["pixel_values"]
        valid = batch["pixel_valid_mask"]
        cached = batch.get("encoder_features")
        if cached is not None:
            # Offline features of the frozen encoder (identical per slice; see data/features.py).
            if any(p.requires_grad for p in self.encoder.parameters()) or getattr(
                    self.encoder, "train_from_layer", None) is not None:
                raise RuntimeError("cached encoder features require a frozen encoder")
            patch_valid = _patch_valid(valid.bool(), tuple(cached.shape[2:4]))
            dense = cached.to(device=images.device, dtype=torch.float32) * patch_valid.unsqueeze(-1)
            bundle = FeatureBundle(dense, [], patch_valid, None, [])
        else:
            bundle = self.encoder(images, valid)
        geometry = [item["geometry"][0] for item in batch["items"]]
        depth_positions = self._depth_positions(
            geometry,
            bundle.dense.shape[1],
            images.device,
        )
        dense = self.adapter(bundle.dense, bundle.valid_mask, depth_positions)
        global_output = self.global_pooler(dense, bundle.valid_mask)

        batch_size, sequence_length = batch["input_ids"].shape
        hidden_size = self.language_model.config.hidden_size
        all_logits = None
        all_hidden = dense.new_zeros((batch_size, sequence_length, hidden_size))
        action_outputs: list[dict[str, Any] | None] = [None] * len(batch["actions"])
        mask_outputs: list[torch.Tensor | None] = [None] * len(batch["actions"])
        usage = HistoryUsage()

        indexed_actions: dict[int, list[tuple[int, dict[str, Any]]]] = {
            index: [] for index in range(batch_size)
        }
        for action_index, action in enumerate(batch["actions"]):
            indexed_actions[action["batch_idx"]].append((action_index, action))

        for batch_index, item in enumerate(batch["items"]):
            registry = EntityRegistry()
            self._region_cache = {}
            selection = choose_dialogue_history(
                self.history_mode,
                self.oracle_fraction,
                self.history_generator,
            )
            detail_cache: dict[int, torch.Tensor] = {}
            dialogue_actions = indexed_actions[batch_index]
            global_tokens = global_output.tokens[batch_index][
                global_output.token_mask[batch_index]
            ]
            dense_b = dense[batch_index : batch_index + 1]
            valid_b = bundle.valid_mask[batch_index : batch_index + 1]
            prepared = self.detail_retriever.prepare(
                dense_b, valid_b, self.detail_retriever.key.out_features)
            labels_b = batch["labels"][batch_index]

            for turn_index, (turn_start, turn_stop) in enumerate(item["turn_boundaries"]):
                detail_cache[turn_index] = self._detail_for_turn(
                    item, turn_index, registry, dense_b, valid_b, images.device, prepared,
                )
                current_actions = [
                    (index, action)
                    for index, action in dialogue_actions
                    if action["turn_index"] == turn_index
                ]
                action_stop = (max(action["seg_token_pos"] for _, action in current_actions) + 1
                               if current_actions else turn_start)

                if self.fused_turn_pass:
                    if not current_actions and not self.compute_text_logits:
                        continue
                    stop = action_stop
                    if self.compute_text_logits:
                        stop = max(self._text_stop(labels_b, turn_start, turn_stop), action_stop)
                    prefix = self._build_prefix(batch, batch_index, stop, turn_index, dense_b, valid_b,
                                                global_tokens, registry, detail_cache)
                    hidden = forward_last_hidden(self.language_model,
                                                 inputs_embeds=prefix.inputs_embeds,
                                                 attention_mask=prefix.attention_mask)
                    if self.compute_text_logits:
                        # LM head only on this turn's positions (the only logits the loss reads).
                        turn_logits = lm_head_logits(self.language_model, hidden[:, turn_start:stop])
                        if all_logits is None:
                            all_logits = dense.new_zeros(
                                (batch_size, sequence_length, turn_logits.shape[-1]))
                        all_logits[batch_index, turn_start:stop] = turn_logits[0]
                    self._decode_actions(hidden, current_actions, batch_index, registry, selection,
                                         dense, bundle, valid, images, geometry, usage,
                                         action_outputs, mask_outputs)
                    continue

                # Reference (unfused) path: [SEG] pass, decode, then a full-turn CE pass.
                if current_actions:
                    action_prefix = self._build_prefix(batch, batch_index, action_stop, turn_index,
                                                       dense_b, valid_b, global_tokens, registry,
                                                       detail_cache)
                    action_forward = forward_hidden(
                        self.language_model,
                        inputs_embeds=action_prefix.inputs_embeds,
                        attention_mask=action_prefix.attention_mask,
                    )
                    self._decode_actions(action_forward.hidden_states[-1], current_actions, batch_index,
                                         registry, selection, dense, bundle, valid, images, geometry,
                                         usage, action_outputs, mask_outputs)
                if not self.compute_text_logits:
                    continue
                final_prefix = self._build_prefix(batch, batch_index, turn_stop, turn_index, dense_b,
                                                  valid_b, global_tokens, registry, detail_cache)
                final_forward = forward_hidden(
                    self.language_model,
                    inputs_embeds=final_prefix.inputs_embeds,
                    attention_mask=final_prefix.attention_mask,
                )
                if all_logits is None:
                    vocabulary_size = final_forward.logits.shape[-1]
                    all_logits = dense.new_zeros(
                        (batch_size, sequence_length, vocabulary_size)
                    )
                all_logits[batch_index, turn_start:turn_stop] = final_forward.logits[
                    0,
                    turn_start:turn_stop,
                ]
                all_hidden[batch_index, turn_start:turn_stop] = final_forward.hidden_states[-1][
                    0,
                    turn_start:turn_stop,
                ]
        self._region_cache = None

        if all_logits is None and not self.compute_text_logits:
            # Placeholder [B,L,1] tied to the graph so "no action" batches still backprop (DDP).
            all_logits = dense.new_zeros((batch_size, sequence_length, 1)) + dense.sum() * 0.0
        if all_logits is None:
            raise ValueError("batch contains no dialogue turns")
        if any(value is None for value in action_outputs):
            raise RuntimeError("not every action was decoded")
        return PRSMedOutput(
            all_logits,
            all_hidden,
            [value for value in mask_outputs if value is not None],
            [value for value in action_outputs if value is not None],
            batch["labels"],
            usage.fractions(),
        )
