"""Qwen3 4B Instruct with a 4-bit LoRA, one adapter per task (D-32a, D-32f).

Each example is a chat: a fixed task instruction with the label list, the input, and the
gold answer in the shared JSON format (D-32b), so ``nlp.metrics`` parses the replies. Loss is
on the answer tokens only. Role training uses every call twice, as written and line-shuffled,
so the model cannot learn "S1 is the agent". Decoding is greedy.
"""

import json
import shutil
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from tuncall.asr.train import IntervalsFromConfig, LoraSettings, ModelRef, TrackerCallback
from tuncall.config import StrictModel
from tuncall.nlp.data import call_turns, dumps
from tuncall.nlp.tasks import DataRef, Example
from tuncall.tracking import Tracker

INSTRUCTIONS = {
    "intent": (
        "Classify the intent of this Tunisian Darija voice-assistant command (Arabic script).\n"
        'Intents: {labels}\nReply with JSON only: {{"intent": "<intent>"}}'
    ),
    "entities": (
        "Extract the entities from this Tunisian Darija voice-assistant command (Arabic "
        "script). Copy each value as written in the text.\nEntity types: {labels}\n"
        'Reply with JSON only: {{"entities": [{{"type": "<type>", "value": "<value>"}}]}}'
    ),
    "sentiment": (
        "Classify the sentiment of this Tunisian Darija comment (Arabic script).\n"
        'Sentiments: {labels}\nReply with JSON only: {{"sentiment": "<sentiment>"}}'
    ),
    "role": (
        "This is a Tunisian Darija customer-support call (Arabic script); its lines may be out "
        "of order. Say which speaker is the agent and which is the customer.\nRoles: {labels}\n"
        'Reply with JSON only: {{"S1": "<role>", "S2": "<role>"}}'
    ),
}
LABELS = "labels.json"  # next to each adapter: the label list its prompt names
MAX_NEW_TOKENS = {"intent": 32, "entities": 256, "sentiment": 16, "role": 32}


class QwenConfig(StrictModel):
    experiment: str
    data: DataRef
    base_model: ModelRef
    quantize_4bit: bool  # false only for the CPU smoke test
    max_length: int  # prompt + answer tokens; longer examples are cut
    shuffle_seed: int  # role: shuffled training copies and the shuffled scoring input
    lora: LoraSettings
    output_root: Path  # adapters go to output_root/<task>
    eval_batch_size: int
    trainer: dict[str, Any]  # TrainingArguments keyword arguments


def chat(
    task: str, labels: list[str], ex: Example, shuffle_seed: int | None = None
) -> tuple[list[dict[str, str]], str, Any]:
    """(messages, gold answer JSON, gold) for one example; role gold is {speaker: role}."""
    if task == "role":
        turns, gold = call_turns(ex.call, shuffle_seed)
        text = role_text(turns)
        answer = dumps(dict(sorted(gold.items())))
    elif task == "entities":
        text, gold = ex.text, ex.gold
        answer = dumps({"entities": [{"type": t, "value": v} for t, v in gold]})
    else:
        text, gold = ex.text, ex.gold
        answer = dumps({task: gold})
    return messages(task, labels, text), answer, gold


def role_text(turns: list[tuple[str, str]]) -> str:
    return "\n".join(f"{s}: {t}" for s, t in turns)


def messages(task: str, labels: list[str], text: str) -> list[dict[str, str]]:
    """The prompt: the task instruction with its label list, then the input."""
    system = INSTRUCTIONS[task].format(labels=", ".join(labels))
    return [{"role": "system", "content": system}, {"role": "user", "content": text}]


class ChatCollator:
    """Prompt + answer token ids; labels are -100 on the prompt and on padding."""

    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tok, self.max_length = tokenizer, max_length

    def encode(self, messages: list[dict[str, str]], answer: str) -> tuple[list[int], int]:
        prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        full = self.tok.apply_chat_template(
            [*messages, {"role": "assistant", "content": answer}], tokenize=False
        )
        assert full.startswith(prompt), "chat template must extend the generation prompt"
        n_prompt = len(self.tok(prompt, add_special_tokens=False).input_ids)
        ids = self.tok(full, add_special_tokens=False).input_ids[: self.max_length]
        return ids, n_prompt

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = [self.encode(b["messages"], b["answer"]) for b in batch]
        width = max(len(ids) for ids, _ in encoded)
        pad = self.tok.pad_token_id
        input_ids = torch.full((len(batch), width), pad)
        labels = torch.full((len(batch), width), -100)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, (ids, n_prompt) in enumerate(encoded):
            input_ids[row, : len(ids)] = torch.tensor(ids)
            mask[row, : len(ids)] = 1
            labels[row, n_prompt : len(ids)] = torch.tensor(ids[n_prompt:])
        return {"input_ids": input_ids, "attention_mask": mask, "labels": labels}


def load_base(cfg: QwenConfig, path: str | None = None) -> tuple[Any, Any]:
    """Base model (4-bit unless disabled) and tokenizer; ``path`` overrides the Hub id."""
    ref = cfg.base_model
    kwargs: dict[str, Any] = {"revision": ref.revision} if path is None else {}
    tok = AutoTokenizer.from_pretrained(path or ref.id, **kwargs)
    dtype = torch.bfloat16 if cfg.trainer.get("bf16") else torch.float32
    if cfg.quantize_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(path or ref.id, dtype=dtype, **kwargs)
    return model, tok


def train_qwen(
    cfg: QwenConfig,
    task: str,
    labels: list[str],
    train: list[Example],
    tracker: Tracker,
    base_path: str | None = None,
) -> tuple[Any, Any]:
    """Train the task's adapter (resuming if a checkpoint exists); return (model, tokenizer)."""
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    model, tok = load_base(cfg, base_path)
    out = cfg.output_root / task
    if (out / "final").is_dir():  # trained earlier; only scoring was left
        (out / "final" / LABELS).write_text(json.dumps(labels) + "\n", encoding="utf-8")
        return PeftModel.from_pretrained(model, str(out / "final")), tok
    if cfg.quantize_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    lora = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    rows = []
    for ex in train:
        for seed in [None, cfg.shuffle_seed] if task == "role" else [None]:
            messages, answer, _ = chat(task, labels, ex, seed)
            rows.append({"messages": messages, "answer": answer})
    args = TrainingArguments(
        output_dir=str(out), report_to="none", remove_unused_columns=False, **cfg.trainer
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=rows,  # pyright: ignore[reportArgumentType]
        data_collator=ChatCollator(tok, cfg.max_length),
        callbacks=[IntervalsFromConfig(), TrackerCallback(tracker)],
    )
    trainer.train(resume_from_checkpoint=any(out.glob("checkpoint-*")) or None)
    trainer.save_model(str(out / "final"))
    (out / "final" / LABELS).write_text(json.dumps(labels) + "\n", encoding="utf-8")
    for ckpt in out.glob("checkpoint-*"):  # optimizer state; the run is finished
        shutil.rmtree(ckpt)
    return model, tok


@torch.no_grad()
def generate(
    model: Any, tok: Any, prompts: list[list[dict[str, str]]], max_new_tokens: int, batch: int
) -> list[str]:
    """Greedy replies, batched with left padding."""
    model.eval()
    tok.padding_side = "left"
    replies = []
    for i in range(0, len(prompts), batch):
        texts = [
            tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            for m in prompts[i : i + batch]
        ]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = enc.to(model.device)
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id
        )
        replies += tok.batch_decode(out[:, enc.input_ids.shape[1] :], skip_special_tokens=True)
    return replies
