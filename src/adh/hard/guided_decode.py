"""Token-guided decoding for stubborn sentences (AP-inspired)."""

from __future__ import annotations

import os
from typing import Callable

from adh.exceptions import HardModeUnavailableError
from adh.hard.adapters import DetectorScoreAdapter


def hard_mode_available() -> bool:
    if os.environ.get("ADH_DISABLE_HARD_MODE") == "1":
        return False
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


def _default_model_id() -> str:
    return os.environ.get("ADH_HARD_LM", "Qwen/Qwen2.5-0.5B-Instruct")


def pick_token_by_detector(
    *,
    token_ids: list[int],
    token_probs: list[float],
    prefix_text: str,
    decode_token: Callable[[int], str],
    detector_scores: Callable[[list[str]], list[float]],
) -> int:
    if len(token_ids) == 1:
        return token_ids[0]
    texts = [prefix_text + decode_token(token_id) for token_id in token_ids]
    scores = detector_scores(texts)
    idx = min(range(len(scores)), key=lambda index: scores[index])
    return token_ids[idx]


def guided_decode_text(
    prompt: str,
    *,
    detector_adapter: DetectorScoreAdapter,
    model_id: str | None = None,
    top_k: int = 16,
    max_new_tokens: int = 64,
) -> str:
    if not hard_mode_available():
        raise HardModeUnavailableError(
            "hard mode requires torch and transformers. "
            "Install extras: pip install 'adversarial-detector-humanizer[hard]'"
        )
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = model_id or _default_model_id()
    tokenizer = AutoTokenizer.from_pretrained(resolved)
    model = AutoModelForCausalLM.from_pretrained(resolved)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated: list[int] = []
    adapter = detector_adapter

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits.float(), dim=-1)[0]
        top_probs, top_ids = torch.topk(probs, k=min(top_k, probs.shape[-1]))
        token_ids = top_ids.cpu().tolist()
        token_prob_values = top_probs.cpu().tolist()
        prefix = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
        chosen = pick_token_by_detector(
            token_ids=token_ids,
            token_probs=token_prob_values,
            prefix_text=prefix,
            decode_token=lambda token_id: tokenizer.decode([token_id]),
            detector_scores=adapter.get_scores,
        )
        if tokenizer.eos_token_id is not None and chosen == tokenizer.eos_token_id:
            break
        generated.append(chosen)
        next_token = torch.tensor([[chosen]], device=device)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(generated, skip_special_tokens=True).strip()
