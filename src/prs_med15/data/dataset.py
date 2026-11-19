"""Dialogue-level loading with explicit slot and action/token alignment."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Sequence
import warnings

import nibabel as nib
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..protocol.serializer import build_memory_message, build_user_message
from ..protocol.tokens import DETAIL_SLOT, GLOBAL_SLOT, REGION_SLOT
from ..models.global_tokens import global_token_count
from .geometry import (TransformRecord, VolumeTransformRecord, apply_transform,
                       apply_volume_transform)
from .preprocess import (PreprocessConfig, load_rgb, preprocess_native_2d,
                         preprocess_volume, ProcessedCache)
from .features import feature_path, load_feature_rows
from .schema import Dialogue, Scan
from .sampling import centre_slab, training_visibility_slab


@dataclass
class DatasetConfig:
    max_len: int = 4096
    image_size: int = 512
    d_max: int = 32
    pad_token_id: int | None = None
    scan_loader: Callable[[Scan], tuple[torch.Tensor, torch.Tensor, Any]] | None = None
    mask_loader: Callable[[Dialogue, str], torch.Tensor | None] | None = None
    global_budget: int = 512
    detail_tokens: int = 64
    region_tokens_per_entity: int = 4
    epoch: int = 0
    seed: int = 0
    eval_slab_policy: str = "centre"
    processed_root: str = "data/processed"
    use_cache: bool = True
    target_inplane_spacing_mm: float | None = None
    target_depth_spacing_mm: float | None = None
    max_native_depth_spacing_mm: float = 5.0
    # Optional offline frozen-encoder feature cache (see data/features.py); None = encode online.
    feature_root: str | None = None


_CACHE_MISS_WARNED = False


def _ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    # transformers BatchEncoding is a Mapping but not a dict subclass.
    from collections.abc import Mapping
    if isinstance(encoded, Mapping) or hasattr(encoded, "input_ids"):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return list(encoded)


def _chat_user_ids(tokenizer: Any, text: str) -> list[int]:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        value = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True)
        from collections.abc import Mapping
        if isinstance(value, Mapping) or hasattr(value, "input_ids"):
            value = value["input_ids"]
        if hasattr(value, "tolist"):
            value = value.tolist()
        if value and isinstance(value[0], list):
            value = value[0]
        ids = [int(v) for v in value]
        # The template emits BOS before every user turn; Mistral's multi-turn format has a single
        # BOS at sequence start, which __getitem__ prepends once.
        bos = getattr(tokenizer, "bos_token_id", None)
        if bos is not None and ids and ids[0] == bos:
            ids = ids[1:]
        return ids
    return _ids(tokenizer, f"[INST] {text} [/INST]")


def _assistant_ids(tokenizer: Any, text: str) -> list[int]:
    ids = _ids(tokenizer, text)
    eos = getattr(tokenizer, "eos_token_id", None)
    return ids + ([int(eos)] if eos is not None else [])


def _preprocess_config(cfg: DatasetConfig) -> PreprocessConfig:
    return PreprocessConfig(
        image_size=cfg.image_size,
        target_inplane_spacing_mm=cfg.target_inplane_spacing_mm,
        target_depth_spacing_mm=cfg.target_depth_spacing_mm,
        max_native_depth_spacing_mm=cfg.max_native_depth_spacing_mm,
        d_max=cfg.d_max,
    )


def _default_scan_loader(
    scan: Scan,
    cfg: DatasetConfig,
    source: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, Any]:
    global _CACHE_MISS_WARNED
    path = Path(scan.image_path)
    if path.suffix == ".npz":
        with np.load(path) as data:
            image = torch.from_numpy(data["image"]).float()
            valid = torch.from_numpy(data["valid_mask"]).bool()
            return image, valid, data.get("geometry", None)
    if scan.input_kind == "native_2d":
        rgb = load_rgb(path)
        image, valid, transform = preprocess_native_2d(rgb, cfg.image_size)
        geometry = dict(scan.geometry_ref) if isinstance(scan.geometry_ref, dict) else {}
        geometry.update({"unknown_geometry": True, "transform": transform.to_dict()})
        return torch.from_numpy(image), torch.from_numpy(valid), geometry
    preprocess_config = _preprocess_config(cfg)
    if cfg.use_cache and source:
        cache = ProcessedCache(Path(cfg.processed_root) / source)
        processed = cache.load(path, preprocess_config)
        if processed is not None:
            cache_path = cache.path_for(path, preprocess_config)
            return (
                torch.from_numpy(processed.image),
                torch.from_numpy(processed.valid_mask),
                {
                    "affine": processed.transform.canonical_affine,
                    "transform": processed.transform.to_dict(),
                    "spacing_mm": processed.spacing_mm,
                    "cache_path": str(cache_path),
                },
            )
        if not _CACHE_MISS_WARNED:
            warnings.warn(
                "processed scan cache is missing; falling back to raw NIfTI preprocessing",
                RuntimeWarning,
                stacklevel=2,
            )
            _CACHE_MISS_WARNED = True
    source_nii = nib.load(str(path))
    nii = nib.as_closest_canonical(source_nii)
    xyz = np.asarray(nii.dataobj)
    dhw = np.transpose(xyz, (2, 1, 0))
    spacing_xyz = nii.header.get_zooms()[:3]
    processed = preprocess_volume(
        dhw,
        tuple(reversed(spacing_xyz)),
        scan.modality,
        preprocess_config,
        source_affine=source_nii.affine,
        canonical_affine=nii.affine,
    )
    return torch.from_numpy(processed.image), torch.from_numpy(processed.valid_mask), {
        "affine": nii.affine.tolist(), "transform": processed.transform.to_dict(),
        "spacing_mm": processed.spacing_mm,
    }


def _default_mask_loader(dialogue: Dialogue, entity_id: str) -> torch.Tensor | None:
    obj = dialogue.objects.get(entity_id)
    if obj is None:
        return None
    if obj.mask_path:
        path = Path(obj.mask_path)
        if path.suffix.lower() == ".png":
            return torch.from_numpy(np.asarray(Image.open(path)) > 0).unsqueeze(0)
        if path.suffix == ".npy":
            return torch.from_numpy(np.load(path).astype(bool))
        if path.name.endswith(".nii.gz"):
            arr = np.asarray(nib.as_closest_canonical(nib.load(str(path))).dataobj) > 0
            return torch.from_numpy(np.transpose(arr, (2, 1, 0)).copy())
    scan = dialogue.scans[0]
    if obj.label_id is not None and scan.label_ref and Path(scan.label_ref).is_file():
        arr = np.asarray(nib.as_closest_canonical(nib.load(scan.label_ref)).dataobj) == obj.label_id
        return torch.from_numpy(np.transpose(arr, (2, 1, 0)).copy())
    if obj.mask_file and scan.label_ref:
        arr = np.asarray(nib.as_closest_canonical(nib.load(str(Path(scan.label_ref) / obj.mask_file))).dataobj) > 0
        return torch.from_numpy(np.transpose(arr, (2, 1, 0)).copy())
    return None


def _cached_entity_masks(
    dialogue: Dialogue,
    cache_path: str,
) -> dict[str, torch.Tensor]:
    """Load a cache archive once and resolve every dialogue entity from it."""
    output: dict[str, torch.Tensor] = {}
    with np.load(cache_path, allow_pickle=False) as value:
        label = value["label"]
        metadata = json.loads(str(value["metadata"])) if "metadata" in value else {}
        keys = metadata.get("object_keys", [])
        packed = value.get("object_masks_packed")
        shape_value = value.get("object_mask_shape")
        shape = tuple(int(part) for part in shape_value) if shape_value is not None else ()
        count = int(np.prod(shape)) if shape else 0
        for entity_id, obj in dialogue.objects.items():
            mask = None
            mask_key = obj.mask_file or obj.mask_path or obj.object_key
            if mask_key in keys and packed is not None:
                index = keys.index(mask_key)
                mask = np.unpackbits(packed[index], count=count).reshape(shape).astype(bool)
            elif obj.label_id is not None and label.size:
                mask = label == int(obj.label_id)
            if mask is not None:
                output[entity_id] = torch.from_numpy(mask)
    return output


class DialogueDataset(Dataset):
    def __init__(self, manifest_rows: Sequence[Dialogue | dict[str, Any] | str], tokenizer: Any,
                 config: DatasetConfig | dict[str, Any] | None = None, mode: str = "train") -> None:
        self.rows = list(manifest_rows)
        self.tokenizer = tokenizer
        self.config = config if isinstance(config, DatasetConfig) else DatasetConfig(**(config or {}))
        self.mode = mode
        self.placeholder_ids = {
            "global": self._single_token_id(GLOBAL_SLOT),
            "detail": self._single_token_id(DETAIL_SLOT),
            "region": self._single_token_id(REGION_SLOT),
        }
        self.seg_token_id = self._single_token_id("[SEG]")

    def _single_token_id(self, token: str) -> int:
        ids = _ids(self.tokenizer, token)
        if len(ids) != 1:
            raise ValueError(f"tokenizer must encode {token!r} as one explicit token, got {ids}")
        return ids[0]

    def __len__(self) -> int:
        return len(self.rows)

    def _dialogue(self, index: int) -> Dialogue:
        row = self.rows[index]
        if isinstance(row, Dialogue):
            return row
        return Dialogue.from_json(row)

    @staticmethod
    def _expand(ids: list[int], token_id: int, count: int) -> list[int]:
        return [value for token in ids for value in ([token] * count if token == token_id else [token])]

    def _serialize_turn(
        self,
        dialogue: Dialogue,
        turn_index: int,
        *,
        first_in_window: bool = False,
        retained_start: int = 0,
        global_count: int = 16,
    ) -> tuple[list[int], list[int], list[tuple[int, Any]], list[dict[str, Any]]]:
        turn = dialogue.turns[turn_index]
        memory_ids: list[int] = []
        memory_records: list[dict[str, Any]] = []
        creators = {action.entity_id: i for i, prior in enumerate(dialogue.turns[:turn_index])
                    for action in prior.actions if action.op in {"create", "update"}}
        for entity_id in turn.reference_entity_ids:
            # Post-action memory already exists in-window. Re-emit only when its creator was dropped.
            if entity_id not in creators or creators[entity_id] >= retained_start:
                continue
            memory = _ids(self.tokenizer, build_memory_message(turn.scan_id, entity_id))
            memory = self._expand(
                memory,
                self.placeholder_ids["region"],
                self.config.region_tokens_per_entity,
            )
            start = len(memory_ids)
            memory_ids.extend(memory)
            positions = [
                start + position
                for position, token in enumerate(memory)
                if token == self.placeholder_ids["region"]
            ]
            memory_records.append({
                "entity_id": entity_id,
                "scan_id": turn.scan_id,
                "positions": positions,
                "turn_index": turn_index,
            })
        user = build_user_message(turn.user_text, first_turn=first_in_window, scan_id=turn.scan_id)
        user_ids = _chat_user_ids(self.tokenizer, user)
        user_ids = self._expand(user_ids, self.placeholder_ids["global"], global_count)
        user_ids = self._expand(user_ids, self.placeholder_ids["detail"], self.config.detail_tokens)
        user_ids = memory_ids + user_ids
        assistant_ids = _assistant_ids(self.tokenizer, turn.assistant_text)
        input_ids = user_ids + assistant_ids
        labels = [-100] * len(user_ids) + assistant_ids.copy()
        seg_positions = [i + len(user_ids) for i, token in enumerate(assistant_ids) if token == self.seg_token_id]
        mask_actions = [a for a in turn.actions if a.op in {"create", "update"}]
        if len(seg_positions) != len(mask_actions):
            raise ValueError(f"turn {turn.turn_id}: SEG/action token mismatch")
        mapped = list(zip(seg_positions, mask_actions, strict=True))
        # Region-result messages are internal inputs and therefore never CE targets.
        for action in mask_actions:
            memory = _ids(self.tokenizer, build_memory_message(action.scan_id, action.entity_id))
            memory = self._expand(memory, self.placeholder_ids["region"], self.config.region_tokens_per_entity)
            start = len(input_ids)
            input_ids.extend(memory)
            labels.extend([-100] * len(memory))
            positions = [
                start + position
                for position, token in enumerate(memory)
                if token == self.placeholder_ids["region"]
            ]
            memory_records.append({
                "entity_id": action.entity_id,
                "scan_id": action.scan_id,
                "positions": positions,
                "turn_index": turn_index,
            })
        return input_ids, labels, mapped, memory_records

    def __getitem__(self, index: int) -> dict[str, Any]:
        dialogue = self._dialogue(index)
        scan_loader = self.config.scan_loader
        scans, valid_masks, geometry = [], [], []
        for scan in dialogue.scans:
            if scan_loader is None:
                image, valid, geo = _default_scan_loader(scan, self.config, dialogue.source)
            else:
                image, valid, geo = scan_loader(scan)
            scans.append(image.float())
            valid_masks.append(valid.bool())
            geometry.append(geo)
        # Load each needed object once, align to the pre-slab grid, then select a slab.
        mask_loader = self.config.mask_loader or _default_mask_loader
        entity_masks: dict[str, torch.Tensor | None] = {}
        cached_masks: dict[str, torch.Tensor] = {}
        if self.config.mask_loader is None and geometry:
            cache_path = geometry[0].get("cache_path") if isinstance(geometry[0], dict) else None
            if cache_path:
                cached_masks = _cached_entity_masks(dialogue, cache_path)
        for turn in dialogue.turns:
            for action in turn.actions:
                if action.op in {"create", "update"} and action.entity_id not in entity_masks:
                    if action.entity_id in cached_masks:
                        entity_masks[action.entity_id] = cached_masks[action.entity_id]
                    else:
                        entity_masks[action.entity_id] = mask_loader(dialogue, action.entity_id)
        features: list[torch.Tensor | None] = []
        for si, scan in enumerate(dialogue.scans):
            depth = scans[si].shape[1]
            aligned = []
            for entity_id, target in list(entity_masks.items()):
                if target is None:
                    continue
                target = target.bool()
                wanted = scans[si].shape[1:]
                if tuple(target.shape) != tuple(wanted):
                    geo = geometry[si]
                    transform_value = geo.get("transform") if isinstance(geo, dict) else None
                    if isinstance(transform_value, dict) and transform_value.get("kind") == "volume":
                        transformed = apply_volume_transform(
                            target.numpy(),
                            VolumeTransformRecord.from_dict(transform_value),
                            order=0,
                        )
                        target = torch.from_numpy(transformed).bool()
                    else:
                        target = F.interpolate(
                            target[None, None].float(),
                            size=wanted,
                            mode="nearest",
                        )[0, 0].bool()
                entity_masks[entity_id] = target
                aligned.append(target)
            if depth > self.config.d_max:
                if self.mode == "train":
                    start, stop = training_visibility_slab(
                        depth,
                        self.config.d_max,
                        aligned,
                        dialogue.dialogue_id,
                        self.config.epoch,
                        self.config.seed,
                    )
                    policy = "train_gt_visibility"
                else:
                    start, stop = centre_slab(depth, self.config.d_max)
                    policy = "eval_centre"
                scans[si] = scans[si][:, start:stop]
                valid_masks[si] = valid_masks[si][start:stop]
                entity_masks = {
                    key: None if value is None else value[start:stop]
                    for key, value in entity_masks.items()
                }
                base = dict(geometry[si] or {})
                base.update({
                    "slab_start": start,
                    "slab_stop": stop,
                    "original_depth": depth,
                    "input_kind": "slab",
                    "slab_policy": policy,
                })
                geometry[si] = base
            else:
                base = dict(geometry[si] or {})
                base.update({
                    "slab_start": 0,
                    "slab_stop": depth,
                    "original_depth": depth,
                    "input_kind": "volume" if scan.input_kind != "native_2d" else "native_2d",
                })
                geometry[si] = base
            if self.config.feature_root:
                features.append(load_feature_rows(
                    feature_path(self.config.feature_root, dialogue.source, scan.image_path, geometry[si]),
                    geometry[si]["slab_start"], geometry[si]["slab_stop"], depth))
        if features and all(value is not None for value in features):
            # With cached encoder features the model reads pixels only for shape/device: ship one
            # channel instead of three (3x less pinning and host->device traffic).
            scans = [value[:1] for value in scans]
        valid_depth = max(int(value.flatten(1).any(-1).sum()) for value in valid_masks)
        global_count = global_token_count(valid_depth, self.config.global_budget)
        retained_start = 0
        # Re-render after every whole-turn drop: the first retained turn must carry scan context.
        while True:
            pieces = [self._serialize_turn(dialogue, i, first_in_window=(i == retained_start),
                                            retained_start=retained_start, global_count=global_count)
                      for i in range(retained_start, len(dialogue.turns))]
            if sum(len(x[0]) for x in pieces) <= self.config.max_len or len(pieces) <= 1:
                break
            retained_start += 1
        if sum(len(x[0]) for x in pieces) > self.config.max_len:
            raise ValueError("one complete turn exceeds max_len; refusing unsafe truncation")
        bos = getattr(self.tokenizer, "bos_token_id", None)
        all_ids: list[int] = [int(bos)] if bos is not None else []
        all_labels: list[int] = [-100] * len(all_ids)
        boundaries: list[tuple[int, int]] = []
        actions: list[dict[str, Any]] = []
        memory_records: list[dict[str, Any]] = []
        scan_index = {scan.scan_id: i for i, scan in enumerate(dialogue.scans)}
        for local_i, (ids, labels, mapped, local_memory) in enumerate(pieces):
            offset = len(all_ids)
            all_ids.extend(ids)
            all_labels.extend(labels)
            boundaries.append((offset, len(all_ids)))
            turn = dialogue.turns[retained_start + local_i]
            for record in local_memory:
                memory_records.append({
                    **record,
                    "turn_index": local_i,
                    "positions": [offset + position for position in record["positions"]],
                })
            for seg_pos, action in mapped:
                target = None if turn.answerability == "not_in_view" else entity_masks.get(action.entity_id)
                if target is not None:
                    target = target.bool()
                    wanted = scans[scan_index[action.scan_id]].shape[1:]
                    if tuple(target.shape) != tuple(wanted):
                        geo = geometry[scan_index[action.scan_id]]
                        transform_value = geo.get("transform") if isinstance(geo, dict) else None
                        if isinstance(transform_value, dict) and transform_value.get("kind") == "volume":
                            transformed = apply_volume_transform(
                                target.cpu().numpy(), VolumeTransformRecord.from_dict(transform_value), order=0)
                            target = torch.from_numpy(transformed).bool()
                        elif isinstance(transform_value, dict) and transform_value.get("kind") == "inplane":
                            transformed = apply_transform(
                                target.cpu().numpy(), TransformRecord.from_dict(transform_value), order=0)
                            target = torch.from_numpy(transformed).bool()
                        else:
                            target = F.interpolate(
                                target[None, None].float(),
                                size=wanted,
                                mode="nearest",
                            )[0, 0].bool()
                    if not target.any():
                        target = None
                actions.append({
                    "seg_token_pos": offset + seg_pos,
                    "scan_idx": scan_index[action.scan_id],
                    "scan_id": action.scan_id,
                    "entity_id": action.entity_id,
                    "turn_id": turn.turn_id,
                    "turn_index": local_i,
                    "op": action.op,
                    "target": target,
                    "references": list(action.refs),
                    "source": dialogue.source,
                    "dimension": "2d" if scans[scan_index[action.scan_id]].shape[1] == 1 else "3d",
                })
        ids_tensor = torch.tensor(all_ids, dtype=torch.long)
        return {
            "dialogue_id": dialogue.dialogue_id,
            "source": dialogue.source,
            "input_kind": geometry[0].get("input_kind", dialogue.scans[0].input_kind),
            "scans": scans, "valid_masks": valid_masks, "geometry": geometry,
            "input_ids": ids_tensor, "labels": torch.tensor(all_labels, dtype=torch.long),
            "attention_mask": torch.ones_like(ids_tensor, dtype=torch.bool),
            "slot_positions": {name: torch.where(ids_tensor == token)[0] for name, token in self.placeholder_ids.items()},
            "turn_boundaries": boundaries, "actions": actions,
            "memory_records": memory_records,
            # Cached frozen-encoder features per scan, or None (model then encodes online).
            "encoder_features": features if features and all(f is not None for f in features) else None,
            "reference_entity_ids": [
                list(t.reference_entity_ids)
                for t in dialogue.turns[retained_start:]
            ],
            "references": [t.reference_entity_ids for t in dialogue.turns[retained_start:]],
            # Kept separately so retrieval sees the current question, never the serialized answer.
            "question_ids": [torch.tensor(_ids(self.tokenizer, t.user_text), dtype=torch.long)
                             for t in dialogue.turns[retained_start:]],
        }
