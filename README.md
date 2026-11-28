# Carnage-Lite Training

This repository builds a C++ Rocket League PPO trainer using the bundled GigaLearnCPP and RLGymCPP code. The current training settings are defined in `src/main.cpp` and `src/TrainingSchedule.h`; the Colab launcher is `Carnage_colab.ipynb`.

## Current training run

- The executable accepts Phase 1 or Phase 2. Phase selection is manual; it does not advance phases automatically.
- Training is configured for 1v1. The current build rejects 2v2. The default `--mode auto` selects 2v2 after a checkpoint reaches 100 billion timesteps, so use `--mode 1v1` to keep it in 1v1.
- Training stops at 1 billion total timesteps by default. `--max-steps` can set a lower cap; requests above 1 billion are capped at 1 billion.
- The trainer loads the newest numbered checkpoint under its save directory when one exists. Use an empty save directory to start without a checkpoint.

## Enabled phase settings

| Phase | Timesteps per iteration and batch | Minibatch | Policy and critic learning rate | Gamma |
| --- | ---: | ---: | ---: | ---: |
| 1 | 50,000 | 25,000 | 0.0002 | 0.99 |
| 2 | 100,000 | 25,000 | 0.0001 | 0.99 |

The phase reward weights are:

| Reward | Phase 1 | Phase 2 | Details |
| --- | ---: | ---: | --- |
| TouchBall | 50 | 5 | |
| SpeedTowardBall | 5 | 1 | |
| FaceBall | 1 | 0.2 | |
| InAir | 0.15 | 0.1 | |
| BallVelocityTowardGoal | — | 3 | |
| GoalAndConcede | — | 30 | A goal gives +30 to the scoring team and −24 to the conceding team. |

The source contains settings for later phases, but the executable currently rejects phases above 2.

## PPO model and environment

| Setting | Value |
| --- | --- |
| Policy hidden layers | `[2048, 2048, 1024]` |
| Critic hidden layers | `[2048, 2048, 1024]` |
| Activation and normalization | ReLU with layer normalization |
| Optimizer | Adam |
| PPO epochs | 1 |
| Entropy scale | 0.01 |
| Clip range | 0.2 |
| GAE gamma | 0.99 for both enabled phases |
| GAE lambda | 0.95 |
| Return standardization | Enabled |
| Random seed | 123 |

The policy and critic use the same learning rate. The optional `--policy-lr` argument overrides both.

The environment uses `DefaultObsPadded(2)`, tick skip 8, and action delay 7. Phase 1 uses random starting states. Phase 2 uses random states 75% of the time and kickoff states 25% of the time. Episodes end on a goal or `NoTouchCondition(10)`.

The local executable defaults to 256 games and CPU. The Colab notebook overrides these with 2,048 games and CUDA.

## Build and run on Windows

Install CMake 3.18 or later, a C++20 compiler such as Visual Studio C++ Build Tools, and Python with PyTorch installed. CMake enables CUDA support when it finds a CUDA toolkit; without one it builds a CPU-only executable.

```powershell
cmake -S . -B build/windows -DCMAKE_BUILD_TYPE=Release
cmake --build build/windows --config Release --parallel 8
```

Run Phase 1 on CPU:

```powershell
build/windows/Release/Carnage.exe collision_meshes --device cpu --mode 1v1 --phase 1 --games 256 --save-dir build/checkpoints
```

Choose Phase 2 manually after reviewing the current training results by changing `--phase 1` to `--phase 2`. Use `--policy-lr` only when you want to override the configured learning rate for both policy and critic.

The trainer accepts `--device cpu|cuda|auto`, `--mode auto|1v1`, `--phase 1|2`, `--games N`, `--max-steps N`, `--max-iterations N`, `--policy-lr N`, and `--save-dir PATH`. Checkpoints are stored in numbered subdirectories of the save directory.

## Google Colab

Use `Carnage_colab.ipynb` with a GPU runtime; the notebook recommends a T4 and stops if CUDA is unavailable. Its first settings cell defaults to Phase 1, 2,048 games, no learning-rate override, and a 5.4-hour session. It clones or fast-forward-pulls this repository, mounts Drive at `MyDrive/Carnage-Lite`, restores saved checkpoints, builds the CUDA trainer, and launches it with `--device cuda` and `--mode auto`.

The notebook syncs complete checkpoints, policy versions, and CSV files to Drive every 60 seconds. Near the session deadline it requests a graceful trainer shutdown and waits up to 120 seconds before forcing an exit. It retries nonzero trainer exits with backoff and creates one daily checkpoint backup.

## Checkpoints and reports

The trainer retains up to eight numbered training checkpoints and up to 32 policy versions. It evaluates new policy versions with the built-in deterministic skill tracker, starting at rating 0.

- `<save-dir>/<timesteps>/`: learner checkpoints used to resume training.
- `<save-dir>/policy_versions/`: policy snapshots used by the skill tracker.
- `<save-dir>/elo_history.csv`: self-play rating history.
- `<save-dir>/reward_episode_metrics.csv`: per-reward Blue and Orange means per player-step, episode totals per player, and each side's share of absolute reward.
- Colab writes the trainer log to `build/train.log` and stores Drive copies under `MyDrive/Carnage-Lite/checkpoints`.

The Colab runtime settings and local defaults differ where noted above; the PPO architecture, enabled phases, reward weights, and phase learning rates come from the C++ training code.
