# SmolVLA LIBERO Evaluation

Docker wrapper for evaluating the [SmolVLA](https://huggingface.co/HuggingFaceVLA/smolvla_libero) model on [LIBERO](https://libero-project.github.io/) benchmarks using `lerobot-eval`.

## Requirements

- NVIDIA GPU with CUDA support
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Docker Compose v2+
- HuggingFace token (`HF_TOKEN` environment variable)

## Usage

### Docker Compose (recommended)

```bash
export HF_TOKEN=hf_...
docker compose build
docker compose run --rm eval                          # defaults: libero_10, 10 episodes
docker compose run --rm eval --task libero_spatial     # choose a task suite
docker compose run --rm eval --task libero_10 --n-episodes 20 --batch-size 2
```

### run.sh script

```bash
./run.sh [TASK] [N_EPISODES] [BATCH_SIZE]
./run.sh libero_spatial 20 2
```

### CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--task` | `libero_10` | Task suite to evaluate |
| `--n-episodes` | `10` | Episodes per task |
| `--batch-size` | `1` | Parallel environments (reduce if OOM) |
| `--task-ids` | all | Specific task IDs to evaluate |
| `--output-dir` | `/workspace/results` | Results directory |

### Supported tasks

`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`

## Results

Results are saved to `results/` (mounted from the container):

- `results/eval_info.json` — per-task success rates and overall metrics
- `results/videos/` — recorded evaluation episodes

## Project structure

```
├── evaluate.py          # Main evaluation script (CLI, GPU detection, results summary)
├── Dockerfile           # Multi-stage build on nvcr.io/nvidia/pytorch:26.02-py3
├── docker-compose.yml   # GPU passthrough, HF cache volume, results mount
├── run.sh               # Convenience wrapper around docker compose
└── results/             # Evaluation output (gitignored)
```
