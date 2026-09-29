import argparse
import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset


def text(row, *names):
    """Pick the first non-empty field from a few common preference-dataset names."""
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="eridai/russian_dpo_qa")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="data/russian_dpo_500.jsonl")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(ds)), min(args.n, len(ds)))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with out.open("w", encoding="utf-8") as f:
        for i in indices:
            row = dict(ds[int(i)])
            prompt = text(row, "prompt", "question", "input", "instruction")
            chosen = text(row, "chosen", "accepted", "response_j", "answer")
            rejected = text(row, "rejected", "rejected_response", "response_k")
            # Some public datasets have extra columns or a few odd rows. I keep only
            # rows that look like plain DPO triples so Soup gets one predictable shape.
            if not (prompt and chosen and rejected):
                continue
            f.write(json.dumps(
                {"prompt": prompt, "chosen": chosen, "rejected": rejected},
                ensure_ascii=False,
            ) + "\n")
            kept += 1

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"source_dataset={args.dataset}")
    print(f"source_rows={len(ds)}")
    print(f"sample_seed={args.seed}")
    print(f"written_rows={kept}")
    print(f"output={out}")
    print(f"sha256={digest}")
    if kept < args.n:
        raise SystemExit(f"only wrote {kept} rows; expected {args.n}")


if __name__ == "__main__":
    main()
