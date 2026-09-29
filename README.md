# Soup Task

Verdict: DON'T SHIP.

The run reached the Soup training setup screen, but I found no evidence that DPO training actually started or changed the model. The GPU stayed idle, no adapter files were saved, and the verification script failed.

See reports/final_report.md for the full report.

## Files

- `soup_takehome_RUN_ALL.ipynb` - Colab notebook used for the run
- `soup.yaml` - training config
- `scripts/create_dataset.py` - dataset builder
- `scripts/verify_training.py` - verification check
- `logs/` - raw timestamped logs and nvidia-smi output
- `verification/verify_training.json` - final verification result
- `reports/final_report.md` - final report
