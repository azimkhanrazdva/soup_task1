# Soup AI Engineer Take-Home Report

## Verdict

**DON'T SHIP.**

The run reached the training setup screen, but I found no evidence that training started or changed the model. I stopped it after about 27 minutes. The GPU stayed idle, the log showed no progress after setup, the output directory had no adapter files, and my verification script failed.

The strongest evidence:

- `soup doctor` passed and the T4 was visible.
- `soup profile` estimated 4.07 GB total memory for the configured run.
- `soup train` printed `Training Setup` at `2026-09-29T11:05:12Z`.
- From `11:31:03Z` through `11:32:02Z`, `nvidia-smi` stayed at `3 MiB / 15360 MiB`, `0%` GPU utilization.
- `output_file_count` was `0`.
- `adapter_files` was empty.
- `passes_minimum_change_check` was `false`.

I treat this as a failed run.

## Memory Budget

The available GPU was a Tesla T4 with 15,360 MiB shown by `nvidia-smi`, reported by PyTorch as about 14.56 GiB.

Soup's profile estimate for `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` with 4-bit LoRA DPO, batch size 1, and sequence length 4096:

| Component | Estimate |
|---|---:|
| Quantized model weights | 0.75 GB |
| LoRA adapter | 0.02 GB |
| Optimizer / adapter states | 0.05 GB |
| Activations | 1.75 GB |
| Runtime overhead | 1.50 GB |
| **Total** | **4.07 GB** |

This estimate fits inside T4 memory. I still expected extra memory from the CUDA context, allocator fragmentation, temporary logits, dataloader/tokenizer buffers, and DPO policy/reference work. A real run should use several GiB of VRAM. This run stayed at 3 MiB, so the gap is not normal overhead. The model never loaded onto the GPU.

## Did the Model Train?

No.

Loss would not prove training by itself. A DPO loop can print metrics while using malformed data, update no trainable adapter, skip batches, or save nothing. I used a separate verification script. It checked the dataset row count and hash, scanned the output directory, looked for adapter artifacts, hashed adapter files, and tried a small preference-margin probe when artifacts existed.

The verification output showed:

```json
{
  "output_dir_exists": true,
  "output_file_count": 0,
  "adapter_files": [],
  "data_rows": 500,
  "passes_minimum_change_check": false
}
```

That check would not prove quality if it passed. It would miss bad labels, data leakage, a wrong reference-model pass, and overfitting. In this run it caught a simpler failure: Soup wrote no adapter, so there was no trained model to ship.

## Silent Failures

### Environment mismatch

`soup doctor` warned that `soup` used `/content/soup-env/bin/python`, while `python` on PATH pointed to `/usr/local/bin/python`. I used the soup interpreter explicitly for checks and scripts. The required train packages were installed in the soup env: `torch`, `transformers`, `trl`, `peft`, `datasets`, `bitsandbytes`, and `accelerate`.

### Missing preflight command

The task asked to use pre-flight checks, but `soup-cli==0.75.1` did not expose a `preflight` command:

```text
No such command 'preflight'.
```

I kept this as raw evidence. I used `soup doctor`, config validation/profile, `soup ship`, and the custom verification script as substitutes.

### Setup banner without training

The main silent failure: `soup train` printed a valid-looking setup banner and then made no visible progress. The GPU monitor stayed idle for the whole observed window. A setup-only check would miss this.

### Empty output

The output directory existed, but it had zero files. Soup's setup checks did not catch that because the failure happened after train started. My verification script caught it and failed the run.

## What Must Change Before Shipping

I would not ship this run until all of these are true:

- `soup train` logs actual steps, loss/progress, and a completed save.
- `nvidia_smi_training.csv` shows real GPU activity and memory use.
- `output/` contains non-empty adapter artifacts.
- `soup ship` passes after training.
- `verify_training.py` passes and shows at least a small base-vs-adapter behavior shift.
- Soup fails loudly if training stalls after setup with no GPU process, no progress, and no artifacts.

## What Surprised Me / What Still Concerns Me

I expected a failed run to throw an error. Instead, the run looked healthy at setup and then sat idle. That worries me because a user can mistake a setup banner for progress.

My remaining concern: even a valid adapter artifact would not prove the model learned the intended behavior. I would add a held-out preference-margin check, sample generations before/after, and a small regression test that fails if `soup train` exits or stalls without saving an adapter.

## AI Tool Use

I used Codex to draft the Colab runner, verification script, and report structure. I accepted the generated scaffold only after checking the raw logs, `nvidia-smi` output, output directory, and verification result. The final verdict comes from those artifacts, not from the tool's suggestion.
