# Carnage-Lite (1.2M)

Fresh-start Rocket League PPO trainer built with the bundled GigaLearnCPP and RLGymCPP code. No weights are imported from the earlier Carnage model.

## Training schedule

- This export trains in 1v1, accepts Phase 1 or Phase 2, and targets 1 billion total timesteps. It stops at the first PPO iteration boundary at or beyond that target. Phase selection remains manual after reviewing gameplay and metrics. The rough allocations are 0.1B for Phase 1 and 0.9B for Phase 2.
- Phases 3–5 and the planned 100B 1v1 / 20B 2v2 segment are paused for this export. The trainer rejects later phases and 2v2 until they are enabled in a future update.
- The observation builder is configured for two teammates from the start. Tick skip is 8 and action delay is 7.
- PPO clip range is GigaLearnCPP's 0.2 default. The trainer reports `SB3 Clip Fraction`; adjust the phase learning rate with `--policy-lr` if you decide to tune toward the guide's approximate 0.08 target. There is no automatic LR controller.

| Phase | Timesteps per iteration | Minibatch | Policy / critic LR | Gamma |
| --- | ---: | ---: | ---: | ---: |
| 1 | 50,000 | 25,000 | 2e-4 | 0.99 |
| 2 | 100,000 | 25,000 | 1e-4 | 0.99 |
| 3 | 150,000 | 50,000 | 1e-4 | 0.995 |
| 4 | 300,000 | 50,000 | 0.8e-4 | 0.998 |
| 5 | 400,000 | 50,000 | 0.8e-4 | 0.998 |

Only Phases 1 and 2 are enabled in this export. The remaining rows are retained as later-stage settings. Advance manually when the bot passes the behavior check; the 0.1B / 0.9B allocations are estimates:

| Active reward | Phase 1 weight | Phase 2 weight | Notes |
| --- | ---: | ---: | --- |
| TouchBall | 50 | 5 | |
| SpeedTowardBall | 5 | 1 | Zero-sum wrapper |
| FaceBall | 1 | 0.2 | |
| InAir | 0.15 | 0.1 | |
| BallVelocityTowardGoal | — | 3 | |
| GoalAndConcede | — | 30 | Scoring team +30; conceding team −24 |

The later behavior checks are:

| Phase | Move on after the bot… |
| --- | --- |
| 1 | hits the ball often and still jumps |
| 2 | regularly pushes the ball toward the opponent's net |
| 3 | takes boost, makes saves, and sometimes connects with aerials |
| 4 | dribbles, flicks, and wins challenges |
| 5 | continues refining; compare each new checkpoint's ELO with older versions |

The policy and critic each use `[512, 512, 512]`, two PPO epochs, entropy scale 0.01, and default clip range 0.2. GigaLearnCPP omits `SB3 Clip Fraction` on its first PPO iteration; it appears in subsequent iteration reports. Experience-buffer sizing is not exposed by this version, so no unsupported buffer option is added.

## Local Windows build and smoke run

Install CMake, Visual Studio C++ build tools, and a Python environment with PyTorch. From this folder:

```powershell
cmake -S . -B build/windows-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build/windows-cpu --config Release -j 8
```

The configured build uses CPU because no CUDA toolkit was detected. To run the bounded one-iteration smoke check:

```powershell
build/windows-cpu/Release/Carnage.exe collision_meshes --device cpu --games 8 --phase 1 --max-iterations 1 --save-dir build/smoke-checkpoints
```

For regular local training, omit `--max-iterations`. Resume is automatic from the highest numeric checkpoint folder in `--save-dir`.

## Colab

1. Use the public [Carnage-Lite-Training repository](https://github.com/vfxjamer/Carnage-Lite-Training). Its HTTPS URL is already set in `REPO_URL` in `Carnage_colab.ipynb`.
2. Open the notebook in Colab, select a T4 GPU runtime, then run cells from top to bottom. It mounts Drive at `MyDrive/Carnage-Lite`, restores checkpoints, builds, and trains for up to about 5.4 hours.
3. Set `PHASE` to 1 or 2 in the notebook after reviewing gameplay and logs. Training targets 1B total timesteps and stops at an iteration boundary after reaching it. The process receives SIGTERM at the session boundary, saves at an iteration boundary, syncs checkpoints and CSVs to Drive, then exits.
4. On later sessions the notebook clones/pulls the repo and resumes the latest synced checkpoint. Drive keeps the latest eight numbered training checkpoints and up to 32 policy versions; a dated backup of the latest checkpoint is created once per day.

No W&B key, replay file, or other credential is required. Training logs go to the notebook output and `build/train.log`.

## Outputs

- `build/checkpoints/<timesteps>/`: learner and optimizer checkpoint, retained locally for resume.
- `build/checkpoints/policy_versions/`: self-play policy snapshots used by the ELO tracker.
- `build/checkpoints/elo_history.csv`: one mode-specific ELO row each time a new checkpoint is evaluated. Ratings start at 0 and are relative self-play ratings; no rank labels are inferred.
- `build/checkpoints/reward_episode_metrics.csv`: per-reward Blue/Orange means per player-step, episode totals per player, and each side's share of absolute reward across the episode.

ELO match batches run against older policy versions through GigaLearnCPP's built-in skill tracker. This export records 1v1 ratings; the planned 2v2 mode is paused. External anchors such as Necto or Nexto are not wired into this trainer.

## Validation

The schedule helper test is under `tests/` and builds independently with CMake. The Windows CPU build and the bounded one-iteration training smoke run should both complete before uploading the source.
