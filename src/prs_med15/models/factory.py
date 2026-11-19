"""Validated construction of production and tiny PRS-Med stacks."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any

import torch
from transformers import MistralConfig, MistralForCausalLM

from ..protocol.tokens import DETAIL_SLOT, GLOBAL_SLOT, REGION_SLOT, SLOT_TOKENS, STRUCT_TOKENS
from .detail_retriever import DetailRetriever
from .encoder import MedSamSliceEncoder, TinyConvEncoder
from .global_tokens import GlobalTokenPooler
from .llm_bridge import (
    add_structural_tokens,
    apply_lora,
    install_delta_vocabulary,
    load_language_model,
)
from .mask_decoder import TinyMaskDecoder, load_medsam_mask_decoder
from .model import PRSMed15
from .query_encoder import QueryEncoder
from .region_memory import MaskedRegionPool, RegionTokenizer
from .slice_adapter import SliceAdapter


class TinyTokenizer:
    """Fixed-size deterministic tokenizer for offline CPU integration tests."""

    def __init__(self, vocab_size: int = 512) -> None:
        self.vocab_size = vocab_size
        self.vocab = {"<pad>": 0, "</s>": 1}
        for token in list(STRUCT_TOKENS) + list(SLOT_TOKENS):
            self.vocab[token] = len(self.vocab)
        self.eos_token_id = 1
        self.pad_token_id = 0
        self.chat_template = "tiny"

    def __len__(self) -> int:
        return self.vocab_size

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(
            r"\{(?:GLOBAL|DETAIL|REGION)\}|\[SEG\]|<[^>]+>|S\d+:E\d+|\w+|[^\s]",
            text,
        )

    def _id(self, token: str) -> int:
        if token in self.vocab:
            return self.vocab[token]
        digest = hashlib.sha256(token.encode()).digest()
        first_hashed = len(self.vocab)
        return first_hashed + int.from_bytes(digest[:8], "little") % (
            self.vocab_size - first_hashed
        )

    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return {"input_ids": [self._id(token) for token in self._tokens(text)]}

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        del add_generation_prompt
        text = " ".join(f"[INST] {value['content']} [/INST]" for value in messages)
        return self(text)["input_ids"] if tokenize else text


def _model_config(cfg: Any) -> dict[str, Any]:
    return cfg.model if hasattr(cfg, "model") else cfg.get("model", {})


def _train_config(cfg: Any) -> dict[str, Any]:
    return cfg.train if hasattr(cfg, "train") else cfg.get("train", {})


def _slot_ids(tokenizer) -> dict[str, int]:
    values = {}
    for name, token in (
        ("global", GLOBAL_SLOT),
        ("detail", DETAIL_SLOT),
        ("region", REGION_SLOT),
    ):
        ids = tokenizer(token, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise ValueError(f"tokenizer does not represent {token} with one token")
        values[name] = ids[0]
    return values


def _freeze_policy(model: PRSMed15, stage: str, train: dict[str, Any] | None = None) -> None:
    text_weight = float(((train or {}).get("loss") or {}).get("text", 1.0))
    model.compute_text_logits = text_weight > 0
    model.encoder.requires_grad_(False)
    for parameter in model.language_model.parameters():
        if "lora_" not in getattr(parameter, "_prs_name", ""):
            parameter.requires_grad_(False)
    for name, parameter in model.language_model.named_parameters():
        if "lora_" in name or ".delta" in name:
            parameter.requires_grad_(True)
    if stage not in {"a", "b", "c"}:
        raise ValueError(f"unknown training stage: {stage}")


def _build_tiny(cfg: Any):
    train = _train_config(cfg)
    model_cfg = _model_config(cfg)
    tiny_cfg = model_cfg.get("tiny", {})
    feature_dim = int(tiny_cfg.get("feature_dim", 8))
    hidden_dim = int(tiny_cfg.get("hidden_dim", 32))
    retrieval_dim = int(tiny_cfg.get("retrieval_dim", 16))
    detail_tokens = int(tiny_cfg.get("detail_tokens", 4))
    global_budget = int(tiny_cfg.get("global_budget", 16))
    region_tokens = int(tiny_cfg.get("region_tokens_per_entity", 4))
    tokenizer = TinyTokenizer(int(tiny_cfg.get("vocab_size", 512)))
    language_model = MistralForCausalLM(
        MistralConfig(
            hidden_size=hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=2 * hidden_dim,
            vocab_size=len(tokenizer),
            max_position_embeddings=int(tiny_cfg.get("max_len", 512)),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    )
    stage = str(train.get("stage", "a"))
    if stage in {"b", "c"} and train.get("enable_lora", True):
        language_model = apply_lora(
            language_model,
            r=int(tiny_cfg.get("lora_r", 2)),
            alpha=int(tiny_cfg.get("lora_alpha", 4)),
        )
    model = PRSMed15(
        TinyConvEncoder(feature_dim=feature_dim, patch_size=4),
        SliceAdapter(feature_dim, heads=2, layers=1),
        GlobalTokenPooler(feature_dim, hidden_dim, budget=global_budget),
        QueryEncoder(
            language_model.get_input_embeddings(),
            hidden_dim,
            retrieval_dim,
            slots=detail_tokens,
            layers=1,
            heads=4,
        ),
        DetailRetriever(feature_dim, retrieval_dim, hidden_dim, layers=1, heads=4),
        language_model,
        TinyMaskDecoder(feature_dim, hidden_dim),
        MaskedRegionPool(feature_dim),
        RegionTokenizer(
            feature_dim,
            hidden_dim,
            hidden_dim,
            tokens_per_entity=region_tokens,
        ),
        _slot_ids(tokenizer),
        history_mode=train.get("history_mode", "oracle"),
        oracle_fraction=float(train.get("history", {}).get("oracle_fraction", 0.5)),
    )
    _freeze_policy(model, stage, train)
    return model, tokenizer, _slot_ids(tokenizer)


def _validate_production_paths(model_cfg: dict[str, Any]) -> None:
    encoder = model_cfg.get("encoder", {})
    llm = model_cfg.get("llm", {})
    required = {
        "model.llm.path": llm.get("path"),
        "model.encoder.hf_cache": encoder.get("hf_cache"),
    }
    missing = [name for name, value in required.items() if not value or not Path(value).exists()]
    if missing:
        raise FileNotFoundError(
            "mandatory production paths are missing: " + ", ".join(missing)
        )


def _build_production(cfg: Any):
    model_cfg = _model_config(cfg)
    train = _train_config(cfg)
    _validate_production_paths(model_cfg)
    encoder_cfg = model_cfg["encoder"]
    llm_cfg = model_cfg["llm"]
    language_model, tokenizer = load_language_model(llm_cfg["path"])
    new_ids = add_structural_tokens(tokenizer)
    install_delta_vocabulary(language_model, new_ids)
    # Gradients reach the (frozen or LoRA) LLM's inputs from the new projectors, so activations of
    # all layers would be kept; checkpoint them (non-reentrant: inputs are inputs_embeds).
    checkpointing = train.get(
        "gradient_checkpointing",
        llm_cfg.get("gradient_checkpointing", True),
    )
    if bool(checkpointing):
        language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        language_model.config.use_cache = False
    stage = str(train.get("stage", "a"))
    if stage in {"b", "c"}:
        lora = llm_cfg.get("lora", {})
        language_model = apply_lora(
            language_model,
            r=int(lora.get("r", 16)),
            alpha=int(lora.get("alpha", 32)),
            targets=tuple(lora.get("targets", ("q_proj", "k_proj", "v_proj", "o_proj"))),
        )
    hidden_dim = language_model.config.hidden_size
    feature_dim = int(encoder_cfg.get("feature_dim", 256))
    detail_cfg = model_cfg.get("detail", {})
    retrieval_dim = int(detail_cfg.get("d", 512))
    detail_tokens = int(detail_cfg.get("K", 64))
    region_tokens = int(model_cfg.get("region", {}).get("tokens_per_entity", 4))
    cache_dir = encoder_cfg["hf_cache"]
    model_id = encoder_cfg.get("hf_id", "wanglab/medsam-vit-base")
    model = PRSMed15(
        MedSamSliceEncoder(
            model_id,
            image_size=int(encoder_cfg.get("image_size", 512)),
            microbatch=int(encoder_cfg.get("slice_microbatch", 8)),
            freeze=bool(encoder_cfg.get("freeze", True)),
            cache_dir=cache_dir,
            local_files_only=True,
        ),
        SliceAdapter(
            feature_dim,
            heads=int(model_cfg.get("adapter", {}).get("heads", 8)),
            layers=int(model_cfg.get("adapter", {}).get("layers", 2)),
            alpha_init=float(model_cfg.get("adapter", {}).get("alpha_init", 0.0)),
        ),
        GlobalTokenPooler(
            feature_dim,
            hidden_dim,
            budget=int(model_cfg.get("global", {}).get("budget", 512)),
            cells=tuple(model_cfg.get("global", {}).get("pool", (4, 4))),
        ),
        QueryEncoder(
            language_model.get_input_embeddings(),
            hidden_dim,
            retrieval_dim,
            slots=detail_tokens,
            layers=int(detail_cfg.get("layers", 2)),
            heads=int(detail_cfg.get("heads", 8)),
        ),
        DetailRetriever(
            feature_dim,
            retrieval_dim,
            hidden_dim,
            layers=int(detail_cfg.get("layers", 2)),
            heads=int(detail_cfg.get("heads", 8)),
        ),
        language_model,
        load_medsam_mask_decoder(
            model_id,
            cache_dir=cache_dir,
            local_files_only=True,
            hidden_dim=hidden_dim,
            sam_dim=feature_dim,
            microbatch=int(encoder_cfg.get("slice_microbatch", 8)),
            extra_tokens=int((model_cfg.get("decoder") or {}).get("extra_tokens", 0)),
            film=bool((model_cfg.get("decoder") or {}).get("film", False)),
        ),
        MaskedRegionPool(feature_dim),
        RegionTokenizer(
            feature_dim,
            hidden_dim,
            hidden_dim,
            tokens_per_entity=region_tokens,
        ),
        _slot_ids(tokenizer),
        history_mode=train.get("history_mode", "oracle"),
        oracle_fraction=float(train.get("history", {}).get("oracle_fraction", 0.5)),
    )
    _freeze_policy(model, stage, train)
    train_from = encoder_cfg.get("train_from_layer")
    if train_from is not None:
        # Partial encoder fine-tuning (2026-11-18 oracle test: frozen features cap 3D Dice ~0.45).
        model.encoder.set_trainable_top(int(train_from), bool(encoder_cfg.get("checkpoint", True)))
    return model, tokenizer, _slot_ids(tokenizer)


def build_model(cfg: Any, tiny: bool = False):
    """Build without touching large weights until every mandatory path is valid."""
    return _build_tiny(cfg) if tiny else _build_production(cfg)
