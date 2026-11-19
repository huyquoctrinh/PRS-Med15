"""Mistral loading, trainable vocabulary deltas, and continuous-token splicing."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import nn

from ..protocol.tokens import STRUCT_TOKENS, SLOT_TOKENS


def mistral_config_from_llava(raw: dict[str, Any]):
    """Build a MistralConfig from a LLaVA-Med config.json without losing RoPE settings.

    transformers 5.x moved ``rope_theta`` into ``rope_parameters``; a plain key filter silently
    drops it and falls back to theta=10000 (LLaVA-Med v1.5 Mistral uses 1e6).
    """
    from transformers import MistralConfig
    allowed = set(MistralConfig().to_dict()) | {"model_type", "architectures"}
    config = MistralConfig(**{k: v for k, v in raw.items() if k in allowed})
    theta = raw.get("rope_theta", (raw.get("rope_parameters") or {}).get("rope_theta"))
    if theta is not None:
        if hasattr(config, "rope_parameters"):
            params = dict(config.rope_parameters or {})
            params.update({"rope_type": params.get("rope_type", "default"), "rope_theta": float(theta)})
            config.rope_parameters = params
        else:
            config.rope_theta = float(theta)
    effective = (getattr(config, "rope_parameters", None) or {}).get("rope_theta", getattr(config, "rope_theta", None))
    if theta is not None and float(effective) != float(theta):
        raise RuntimeError(f"rope_theta not applied: wanted {theta}, got {effective}")
    return config


def load_language_model(path: str | Path, dtype: torch.dtype = torch.bfloat16):
    """Load language weights strictly while dropping only documented LLaVA prefixes."""
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, MistralConfig, MistralForCausalLM
    root = Path(path)
    raw = json.loads((root / "config.json").read_text())
    allowed = set(MistralConfig().to_dict()) | {"model_type", "architectures"}
    config = mistral_config_from_llava(raw)
    model = MistralForCausalLM(config).to(dtype=dtype)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    wanted = set(model.state_dict())
    state: dict[str, torch.Tensor] = {}
    dropped: set[str] = set()
    allow_prefixes = ("model.vision_tower.", "model.mm_projector.", "vision_tower.", "mm_projector.")
    for shard in sorted(set(index.values())):
        values = load_file(str(root / shard), device="cpu")
        for key, value in values.items():
            if key in wanted:
                state[key] = value
            elif key.startswith(allow_prefixes):
                dropped.add(key)
            else:
                raise RuntimeError(f"unexpected checkpoint key not on drop allow-list: {key}")
    missing = wanted - set(state)
    if missing:
        raise RuntimeError(f"missing language keys ({len(missing)}): {sorted(missing)[:8]}")
    model.load_state_dict(state, strict=True)
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    return model, tokenizer


def add_structural_tokens(tokenizer: Any) -> list[int]:
    before = len(tokenizer)
    tokenizer.add_special_tokens({"additional_special_tokens": list(STRUCT_TOKENS) + list(SLOT_TOKENS)})
    return list(range(before, len(tokenizer)))


class DeltaTokenEmbedding(nn.Module):
    """Frozen original embedding plus independently trainable rows for added IDs."""
    def __init__(self, base: nn.Embedding, new_token_ids: Iterable[int], noise_std: float = 1e-3) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        ids = torch.tensor(list(new_token_ids), dtype=torch.long)
        self.register_buffer("new_token_ids", ids)
        mean = base.weight.detach().float().mean(0, keepdim=True)
        self.delta = nn.Parameter(mean.expand(len(ids), -1).clone() + torch.randn(len(ids), base.embedding_dim) * noise_std)
        self.num_embeddings = max(base.num_embeddings, int(ids.max()) + 1 if len(ids) else 0)
        self.embedding_dim = base.embedding_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        safe = ids.clamp_max(self.base.num_embeddings - 1)
        out = self.base(safe)
        is_new = torch.isin(ids, self.new_token_ids)
        if ((ids >= self.base.num_embeddings) & ~is_new).any():
            raise IndexError("token id is outside base vocabulary and has no delta row")
        # One gather instead of a full-size torch.where per added token (same values).
        order = torch.argsort(self.new_token_ids)
        rows = order[torch.searchsorted(self.new_token_ids[order], ids.clamp_min(0))].clamp_max(len(order) - 1)
        return torch.where(is_new.unsqueeze(-1), self.delta[rows].to(out.dtype), out)

    def delta_state_dict(self) -> dict[str, torch.Tensor]:
        return {"new_token_ids": self.new_token_ids.cpu(), "delta": self.delta.detach().cpu()}

    def load_delta_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        if not torch.equal(self.new_token_ids.cpu(), state["new_token_ids"].cpu()):
            raise ValueError("delta token IDs do not match")
        self.delta.data.copy_(state["delta"].to(self.delta))


class DeltaLMHead(nn.Module):
    def __init__(self, base: nn.Linear, new_token_ids: Iterable[int], initial_rows: torch.Tensor) -> None:
        super().__init__(); self.base = base; self.base.weight.requires_grad_(False)
        if self.base.bias is not None: self.base.bias.requires_grad_(False)
        self.register_buffer("new_token_ids", torch.tensor(list(new_token_ids), dtype=torch.long))
        self.delta = nn.Parameter(initial_rows.detach().clone())
        self.out_features = max(base.out_features, int(self.new_token_ids.max()) + 1 if len(self.new_token_ids) else 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        old = self.base(x)
        out = x.new_full((*x.shape[:-1], self.out_features), torch.finfo(x.dtype).min)
        out[..., :old.shape[-1]] = old
        out[..., self.new_token_ids] = torch.nn.functional.linear(x, self.delta.to(x.dtype))
        return out

    def delta_state_dict(self):
        return {"new_token_ids": self.new_token_ids.cpu(), "delta": self.delta.detach().cpu()}

    def load_delta_state_dict(self, state):
        if not torch.equal(self.new_token_ids.cpu(), state["new_token_ids"].cpu()): raise ValueError("delta token IDs do not match")
        self.delta.data.copy_(state["delta"].to(self.delta))


def install_delta_vocabulary(model: nn.Module, new_token_ids: Iterable[int]):
    ids = list(new_token_ids)
    emb = DeltaTokenEmbedding(model.get_input_embeddings(), ids)
    model.set_input_embeddings(emb)
    head = DeltaLMHead(model.get_output_embeddings(), ids, emb.delta)
    model.set_output_embeddings(head)
    return emb, head


def apply_lora(model: nn.Module, r: int = 16, alpha: int = 32,
               targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")):
    from peft import LoraConfig, get_peft_model
    return get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, target_modules=list(targets),
                                             bias="none", task_type="CAUSAL_LM"))


@dataclass(frozen=True)
class SlotSpan:
    batch: int
    positions: torch.Tensor | list[int]
    key: str


@dataclass
class SplicedInputs:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor | None
    position_map: list[dict[str, Any]]


def build_inputs_embeds(input_ids: torch.Tensor, slot_spans: Iterable[SlotSpan | Mapping[str, Any]],
                        continuous: Mapping[str, torch.Tensor], embedding_table: nn.Module,
                        attention_mask: torch.Tensor | None = None,
                        labels: torch.Tensor | None = None) -> SplicedInputs:
    embeds = embedding_table(input_ids).clone()
    attention = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool().clone()
    aligned = labels.clone() if labels is not None else None
    position_map: list[dict[str, Any]] = []
    offsets: dict[str, int] = {}
    for raw in slot_spans:
        span = raw if isinstance(raw, SlotSpan) else SlotSpan(int(raw["batch"]), raw["positions"], str(raw["key"]))
        pos = torch.as_tensor(span.positions, dtype=torch.long, device=input_ids.device)
        values = continuous[span.key]
        if values.ndim == 3: values = values[span.batch]
        start = offsets.get(span.key, 0)
        values = values[start:start + len(pos)].to(embeds)
        if len(values) != len(pos):
            raise ValueError(f"slot {span.key}: {len(pos)} placeholders but {len(values)} continuous rows remain")
        embeds[span.batch, pos] = values
        if aligned is not None: aligned[span.batch, pos] = -100
        offsets[span.key] = start + len(pos)
        position_map.extend({"batch": span.batch, "position": int(p), "key": span.key,
                             "continuous_index": start+i} for i,p in enumerate(pos.tolist()))
    return SplicedInputs(embeds, attention, aligned, position_map)


def forward_hidden(model: nn.Module, *, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor,
                   labels: torch.Tensor | None = None):
    return model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels,
                 output_hidden_states=True, return_dict=True, use_cache=False)


def _causal_lm_parts(model: nn.Module) -> tuple[nn.Module, nn.Module]:
    """(decoder backbone, LM head) of a (possibly PEFT-wrapped) *ForCausalLM."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base.model, base.get_output_embeddings()


def forward_last_hidden(model: nn.Module, *, inputs_embeds: torch.Tensor,
                        attention_mask: torch.Tensor) -> torch.Tensor:
    """Backbone only: final-norm hidden states [B,L,H]; no LM head, no per-layer hidden list."""
    backbone, _ = _causal_lm_parts(model)
    return backbone(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                    use_cache=False, return_dict=True).last_hidden_state


def lm_head_logits(model: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    return _causal_lm_parts(model)[1](hidden)


def gather_seg_states(hidden: torch.Tensor, seg_positions: Iterable[tuple[int, int]]) -> torch.Tensor:
    pairs = list(seg_positions)
    if not pairs: return hidden.new_empty((0, hidden.shape[-1]))
    b, p = zip(*pairs)
    return hidden[torch.tensor(b,device=hidden.device), torch.tensor(p,device=hidden.device)]


@torch.no_grad()
def generate_with_seg_states(model: nn.Module, seg_token_id: int, *, input_ids: torch.Tensor | None = None,
                             inputs_embeds: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None,
                             embedding_table: nn.Module | None = None, **generate_kwargs):
    """Generate, then run one teacher-forced pass and collect each generated SEG state."""
    if (input_ids is None) == (inputs_embeds is None): raise ValueError("pass exactly one prompt representation")
    prompt_len = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
    generated = model.generate(input_ids=input_ids, inputs_embeds=inputs_embeds,
                               attention_mask=attention_mask, **generate_kwargs)
    sequences = generated.sequences if hasattr(generated, "sequences") else generated
    # HF may return prompt+continuation for input_ids, but continuation-only for inputs_embeds.
    if sequences.shape[1] >= prompt_len and input_ids is not None and torch.equal(sequences[:, :prompt_len], input_ids):
        continuation = sequences[:, prompt_len:]
    else:
        continuation = sequences
    table = embedding_table or model.get_input_embeddings()
    if input_ids is not None:
        full_ids = torch.cat((input_ids, continuation), 1)
        full_embeds = table(full_ids)
    else:
        full_ids = continuation
        full_embeds = torch.cat((inputs_embeds, table(continuation)), 1)
    full_attention = torch.ones(full_embeds.shape[:2], dtype=torch.bool, device=full_embeds.device)
    out = forward_hidden(model, inputs_embeds=full_embeds, attention_mask=full_attention)
    positions = [(b, prompt_len + p) for b in range(continuation.shape[0])
                 for p in torch.where(continuation[b] == seg_token_id)[0].tolist()]
    return sequences, gather_seg_states(out.hidden_states[-1], positions), positions
