# REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning?

This repository is the **official evaluation codebase** for **REI-Bench**, from the paper [**REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning?**](https://jcx0110.github.io/REI-Bench-web/) REI-Bench is an evaluation benchmark for embodied task planners under **referential and coreferential vagueness** in natural language instructions.

## 🤖 Authors

**Chenxi Jiang**, **Chuhao Zhou**, [**Jianfei Yang**](https://marsyang.site/)  
[MARS Lab](https://marslab.tech/), Nanyang Technological University (NTU)

## 🧭 Introduction

We introduce **REI-Bench**, a benchmark for assessing whether embodied agents can interpret **vague, underspecified, or coreferential** human instructions in task planning. In real-world settings, users often give instructions like "put the drink in the fridge" or "turn off the electronic device" without specifying *which* drink or device. This **referential vagueness** is a fundamental bottleneck: the agent must ground language to the right objects and then plan and execute. Existing benchmarks (e.g., ALFRED) largely use clear, unambiguous instructions and thus do not evaluate this capability. REI-Bench fills this gap by providing a controlled benchmark where instruction vagueness is systematically varied, enabling rigorous evaluation and method development.

## ⚙️ Requirements

- **Python**: 3.8+ (tested on Ubuntu 22.04, Python 3.8).
- **PyTorch**: 2.0+ (e.g., `torch==2.0.0`, `torchvision==0.15.1`); install from [pytorch.org](https://pytorch.org/get-started/locally/) according to your CUDA version.
- **Other major dependencies**: `transformers`, `hydra-core`, `omegaconf`, `guidance`, `ai2thor` (for ALFRED).

## Prepare

Evaluation requires your own data; set the `split` and data paths in `configs/config.yaml`.

**Environment**

```bash
conda env create -f requriements.yaml
conda activate reibench
```

**Data (ALFRED)**

```bash
sh data/raw/alfred/download_data.sh json
```

Output is `data/raw/alfred/json_2.1.0`. Point config `split` and data paths to it (and e.g. `data/raw/alfred/splits/` for splits).

**REI-Bench splits (one-time)**

After downloading ALFRED data and placing the source split file (with REI-Bench fields), run once from the project root to generate `data/rei_bench/splits/rei_bench.json`:

```bash
python data/rei_bench/splits/prepare_rei_bench_splits.py
```

If your source split file is elsewhere, specify it:

```bash
python data/rei_bench/splits/prepare_rei_bench_splits.py --input /path/to/source_split.json
```

## Run

From the project root:

```bash
python scripts/evaluate.py
```

Override options via Hydra, e.g. `python scripts/evaluate.py task_difficulty=explicit-standard`.

Without a display (e.g. headless server), use `./scripts/run_evaluate_with_xvfb.sh` (you can append overrides). If you hit GLX errors, install `xvfb libgl1-mesa-glx libgl1-mesa-dri` or use [ai2thor-docker](https://github.com/allenai/ai2thor-docker).

Note: This repository is under active development. If you encounter issues with environment setup, please open an issue.



## 📜 Citation

If you use REI-Bench or TOCC in your work, please cite:

```bibtex
  @inproceedings{jiang2026reibench,
    title={REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning?},
    author={Jiang, Chenxi and Zhou, Chuhao and Yang, Jianfei},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    month={April},
    url = {https://openreview.net/pdf?id=vmBIF25KLf}
  }
  
  @article{jiang2025rei,
  title={REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning?},
  author={Jiang, Chenxi and Zhou, Chuhao and Yang, Jianfei},
  journal={arXiv preprint arXiv:2505.10872},
  year={2025}
  }
```

## Acknowledgement

We thank the [ALFRED](https://askforalfred.com/) team for the benchmark and data, [LoTa-Bench](https://choi-jaewoo.github.io/LoTa-Bench/) for the task-planning evaluation framework, and [AI2-THOR](https://github.com/allenai/ai2thor) for the simulation environment.

