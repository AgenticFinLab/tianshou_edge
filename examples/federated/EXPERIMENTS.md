# Experiment ledger

Every experiment run on the federated platform, with protocol, headline
results, and the location of the raw data. Raw data files stay local
(`results/`, not committed); this ledger is the committed record.

Conventions: per-client evaluation returns are always recorded (never only
pooled means); statistics across runs use seeds as the unit; measurement
caveats (rescaling, co-location) are stated inline.

## E1. DQN multi-seed comparison (CartPole, gravity heterogeneity)

* Protocol: 3 clients (gravity 9.3/9.8/10.3), 100 rounds x (100 env steps +
  10 gradient steps), eval 20 greedy episodes every 2 rounds, 3 seeds.
* Result (final score = per-client mean over last 10 evals, pooled; mean
  +/- std over seeds):

| method | final score |
|---|---|
| local_only | 207.5 +/- 24.1 |
| fedavg | 186.2 +/- 46.5 |
| fedprox (mu=0.05) | 216.1 +/- 79.5 |

* Verdict: all methods learn (random baseline ~10); no aggregation method
  separates from local-only at weak heterogeneity; an earlier single-seed
  FedProx drop was seed noise.
* Data: `results/dqn_fed_multiseed.csv`.

## E2. SAC baseline calibration (Pendulum)

* Protocol: standard Pendulum, SAC [256, 256], lr 1e-3, tau 0.005, fixed
  alpha 0.2, batch 256, UTD=1, 10k random warm-up, 100k env steps, eval 20
  episodes every 5k steps.
* Result: reach -206 by 5k training steps, stable in the -137..-224 band
  (literature-typical for SAC-Pendulum). Found and fixed the twin-critic
  degeneracy (critic2 must be independently initialized).
* Also: local-only 60k runs at g = 8/10/12 all learn (final means -166 /
  -178 / -190).
* Data: `results/sac_baseline_calibration.csv`, `results/sac_local_g*.csv`.

## E3. Step-6 federation comparison (Pendulum, g = 8/10/12)

* Protocol: 4 methods x 3 seeds x 3 clients, unified 60k-step budget,
  UTD=1, eval 20 episodes per round, per-client records.
* Result (final per-client eval return, mean +/- std over seeds):

| method | client0 (g=8) | client1 (g=10) | client2 (g=12) |
|---|---|---|---|
| local_only | -156.7 +/- 14.8 | -177.3 +/- 15.3 | -195.9 +/- 10.7 |
| fedavg_all | -165.4 +/- 0.7 | -182.5 +/- 15.0 | -201.3 +/- 8.2 |
| fedavg_actor_only | -162.5 +/- 17.1 | -169.4 +/- 5.3 | -194.0 +/- 3.1 |
| fedprox_actor | -156.6 +/- 14.5 | -208.3 +/- 50.5 | -220.5 +/- 57.9 |

* Verdict: infrastructure validated; no method separates from local_only
  at this heterogeneity. The one cross-seed-consistent phenomenon (A2
  analysis): FedProx's early-learning slowdown (pooled mean -353/-424/-390
  at 4k/8k/12k steps vs ~-175 for all other methods) without final-return
  payoff.
* Data: `results/sac_fed_s{42,43,44}.csv`, `results/a2_diagnostic_report.md`.

## E4. Laptop serial vs Ray benchmark (PPO CartPole)

* Protocol: 3 clients x 5 rounds x (500 steps + 2 epochs), identical tasks.
* Result: serial 2.6s wall; Ray 0.9s (+14.1s one-time actor build);
  speedup 3.00x; platform comm+coordination overhead 0.00s/round.
* Data: `results/benchmark_serial_vs_ray.md`.

## E5. Server three-mode benchmark, small model (8xA40, 64 cores)

* Protocol: SAC [256, 256] batch 256 on HalfCheetah-v5, 4 clients x 20
  rounds x (1 env step + 1 gradient step, UTD=1), FedAvg. Rescaled from 100
  rounds after the OMP trap was found (see below). serial and ray-gpu ran
  concurrently (co-located measurement); ray-cpu ran after serial.
* Result:

| mode | wall (s) | startup (s) | steady round (s) | gpu util mean |
|---|---|---|---|---|
| serial | 1607.7 | 0.0 | 80.19 +/- 0.67 | - |
| ray-cpu | 393.6 | 10.1 | 19.52 +/- 0.28 | - |
| ray-gpu | 475.8 | 11.5 | 23.76 +/- 0.25 | 88.7% |

* Verdict: Ray parallelism gives 4.08x over serial; **CPU-parallel beats
  GPU-parallel (1.21x)** because the per-update cost is CPU-side Python
  overhead at this model size.
* Data: `results/benchmark_server.md`.

## E6. GPU crossover sweep and GPU-hungry benchmark

* Sweep: ms/update for SAC on HalfCheetah across hidden sizes and batch
  sizes on CPU (OMP=8) vs A40. GPU time is flat (~29ms) until the largest
  cells; CPU time scales with compute. Crossover: [512, 512] x batch 1024
  (GPU 1.85x); decisive at [1024, 1024] x batch 1024 (5.41x).
* GPU-hungry three-mode run at the crossover config ([1024, 1024] x
  batch 1024, 4 clients x 20 rounds):

| mode | wall (s) | steady round (s) | gpu util mean |
|---|---|---|---|
| serial | 12459.6 | 616.84 +/- 3.81 | - |
| ray-cpu | 3300.9 | 164.02 +/- 0.77 | - |
| ray-gpu | 1371.2 | 68.52 +/- 0.37 | 100.0% |

* Verdict: **ray-gpu is 9.08x over serial and 2.41x over ray-cpu**; the
  measured end-to-end crossover confirms the sweep.
* Data: `results/update_cost_sweep.md`,
  `results/benchmark_server_gpu_hungry.md`.

## E7. Long-duration Meta-World federated training (reach-v3, push-v3)

* Protocol: 6 clients per federation, SAC [256, 256] batch 256, UTD=1,
  600 rounds per cycle, eval every 25 rounds; cycles auto-restart with CSV
  backups (watchdog v2).
* Result (mean eval return pooled over clients; 3 complete cycles):

| task | cycle start | cycle end (typical) |
|---|---|---|
| reach-v3 | ~3500 (already learned) | ~3700-3900 plateau |
| push-v3 | ~90-110 | ~3200-3360 (consistent across cycles) |

* Note: occupation-run quality (single seed, fresh init per cycle) — the
  consistent per-cycle endpoints are replication evidence, not final
  experiment data.
* Data: `results/metaworld_reconstructed/*` (local reconstructions),
  server `results/longrun_metaworld_*.csv` and `*_history.jsonl`.

## Cross-cutting findings

* **OMP trap**: with the container's default 64-thread BLAS dispatch, one
  SAC update costs 4051ms; at OMP_NUM_THREADS=8 it costs 14ms (290x). The
  benchmark harness pins OMP=8 for all modes.
* **GPU placement verification**: per-actor `get_device_info` reporting
  (CUDA_VISIBLE_DEVICES, device index, GPU name) — placement is never
  inferred from `num_gpus` alone.
* **Cluster policy compliance**: the container is reclaimed after ~3h
  below 20% container-average GPU utilization; occupation is maintained
  with real federated training (no synthetic burners).
