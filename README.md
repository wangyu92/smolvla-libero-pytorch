# SmolVLA LIBERO Evaluation

Docker wrapper for evaluating the [SmolVLA](https://huggingface.co/HuggingFaceVLA/smolvla_libero) model on [LIBERO](https://libero-project.github.io/) benchmarks using `lerobot-eval`.

## Requirements

- NVIDIA GPU with CUDA support
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Docker Compose v2+
- (선택) HuggingFace token (`HF_TOKEN`) — 모델이 public(Apache 2.0)이므로 토큰 없이도 다운로드 가능. 설정 시 rate limit 완화

## Usage

### Docker Compose (recommended)

```bash
export HF_TOKEN=hf_...  # 선택 사항 — 없어도 동작함
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

LIBERO 벤치마크는 **5개의 task suite**로 구성되며, 각 suite에는 고유한 자연어 instruction을 가진 태스크들이 포함되어 있다.

| Task Suite | 태스크 수 | 설명 |
|---|---|---|
| `libero_spatial` | 10 | 공간 관계 이해 |
| `libero_object` | 10 | 객체 인식 |
| `libero_goal` | 10 | 목표 상태 달성 |
| `libero_10` | 10 | 다양한 태스크 혼합 |
| `libero_90` | 90 | 대규모 태스크 셋 |

각 태스크는 하나의 자연어 instruction(e.g. *"put the bowl on the stove"*)에 대응하며,
모델은 이 instruction을 입력받아 로봇을 조작한다.
평가 시 `--n-episodes`만큼 반복 실행하여 **성공률(success rate)**을 측정한다.

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
