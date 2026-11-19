# GPU smoke test

Run `bash scripts/smoke_gpu.sh` only after limited manifests have been prepared and the configured
MedSAM/LLaVA-Med weights are present locally. The script refuses to start unless `nvidia-smi` finds a
GPU with no compute processes owned by another user. It does not launch background jobs.

The sequence checks text-only logits against the PRS-Med LLaVA fork, runs one native-2D and one 3D
forward while reporting peak allocated memory, performs 200-step eight-example stage-A and stage-B
overfit runs, and finishes with a multi-turn CREATE → REF → referenced CREATE dialogue. Override the
Python executable with `PRSMED15_PYTHON=/path/to/python` if needed. This script is intentionally not
part of the CPU test suite.
