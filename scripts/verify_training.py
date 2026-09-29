import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    """Hash artifacts so the report can point to exact files, not hand-wavy output."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pairs(path, limit):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def score_answer(model, tokenizer, prompt, answer, max_length):
    import torch

    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).input_ids
    full_ids = tokenizer(
        prompt + answer,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).input_ids
    # Score only the answer tokens. The prompt is context, not something the
    # adapter should get credit for predicting.
    answer_start = min(prompt_ids.shape[1], full_ids.shape[1] - 1)
    full_ids = full_ids.to(model.device)
    with torch.no_grad():
        logits = model(full_ids).logits[:, :-1, :]
        labels = full_ids[:, 1:]
        logp = torch.log_softmax(logits, dim=-1)
        token_logp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return float(token_logp[:, answer_start - 1 :].mean().detach().cpu())


def preference_margin_probe(base_model, adapter_dir, data_path, limit, max_length):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # This is intentionally small. I only need to catch "adapter did nothing",
    # not run a full eval suite inside the take-home notebook.
    quant = BitsAndBytesConfig(load_in_4bit=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant,
        device_map="auto",
        trust_remote_code=True,
    )
    pairs = load_pairs(data_path, limit)

    def margins(m):
        values = []
        for row in pairs:
            chosen = score_answer(m, tokenizer, row["prompt"], row["chosen"], max_length)
            rejected = score_answer(m, tokenizer, row["prompt"], row["rejected"], max_length)
            values.append(chosen - rejected)
        return values

    base_margins = margins(model)
    model = PeftModel.from_pretrained(model, adapter_dir)
    adapter_margins = margins(model)
    del model
    torch.cuda.empty_cache()

    shifts = [a - b for a, b in zip(adapter_margins, base_margins)]
    return {
        "pairs": len(pairs),
        "base_mean_margin": sum(base_margins) / len(base_margins),
        "adapter_mean_margin": sum(adapter_margins) / len(adapter_margins),
        "mean_margin_shift": sum(shifts) / len(shifts),
        "max_abs_margin_shift": max(abs(x) for x in shifts),
        "all_shifts": shifts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--data", default="data/russian_dpo_500.jsonl")
    parser.add_argument("--report", default="verification/verify_training.json")
    parser.add_argument("--base-model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--probe-pairs", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--skip-margin-probe", action="store_true")
    args = parser.parse_args()

    out = Path(args.output_dir)
    data = Path(args.data)
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in out.rglob("*") if p.is_file()) if out.exists() else []
    adapter_files = [
        p for p in files
        if p.name in {"adapter_model.safetensors", "adapter_model.bin"}
        or "adapter" in p.name.lower()
    ]

    result = {
        "output_dir_exists": out.exists(),
        "output_file_count": len(files),
        "output_files": [str(p) for p in files],
        "adapter_files": [
            {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in adapter_files
        ],
        "data_rows": sum(1 for _ in data.open("r", encoding="utf-8")) if data.exists() else None,
        "data_sha256": sha256(data) if data.exists() else None,
        "passes_minimum_change_check": False,
        "margin_probe": None,
        "margin_probe_error": None,
        "limitations": [
            "This detects adapter files and a small behavior shift, not model quality.",
            "It cannot prove the reference model pass was correct.",
            "It cannot catch data leakage, bad labels, or reward hacking.",
            "A non-empty adapter can still be trained on the wrong data/config.",
        ],
    }

    result["passes_minimum_change_check"] = (
        result["output_dir_exists"]
        and len(adapter_files) > 0
        and all(item["bytes"] > 1024 for item in result["adapter_files"])
        and result["data_rows"] == 500
    )

    if result["passes_minimum_change_check"] and not args.skip_margin_probe:
        try:
            result["margin_probe"] = preference_margin_probe(
                args.base_model,
                str(out),
                str(data),
                args.probe_pairs,
                args.max_length,
            )
            result["passes_minimum_change_check"] = (
                result["passes_minimum_change_check"]
                and result["margin_probe"]["max_abs_margin_shift"] > 1e-5
            )
        except Exception as exc:
            result["margin_probe_error"] = repr(exc)
            result["passes_minimum_change_check"] = False

    report.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not result["passes_minimum_change_check"]:
        raise SystemExit("verification failed: no convincing adapter artifact or behavior change")


if __name__ == "__main__":
    main()
