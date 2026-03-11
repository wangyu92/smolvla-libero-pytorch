# SmolVLA LIBERO Evaluation

Docker wrapper for evaluating the [SmolVLA](https://huggingface.co/HuggingFaceVLA/smolvla_libero) model on [LIBERO](https://libero-project.github.io/) benchmarks using `lerobot-eval`.

## Requirements

- NVIDIA GPU with CUDA support
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Docker Compose v2+
- (Optional) HuggingFace token (`HF_TOKEN`) — the model is public (Apache 2.0), so it can be downloaded without a token. Setting one helps avoid rate limits

## Usage

### Docker Compose (recommended)

```bash
export HF_TOKEN=hf_...  # optional — works without it
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

### LIBERO benchmark structure

The LIBERO benchmark consists of **5 task suites**, each containing tasks with unique natural language instructions.

| Task Suite | # Tasks | Description |
|---|---|---|
| `libero_spatial` | 10 | Spatial relation understanding |
| `libero_object` | 10 | Object recognition |
| `libero_goal` | 10 | Goal state achievement |
| `libero_10` | 10 | Mixed task variety |
| `libero_90` | 90 | Large-scale task set |

Each task corresponds to a single natural language instruction (e.g. *"put the bowl on the stove"*),
and the model receives this instruction as input to control the robot.
During evaluation, each task is run `--n-episodes` times to measure the **success rate**.

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
