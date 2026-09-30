# AdaTrust — Code

Dual-loop AI-driven adaptive consensus: BayesElect (Bayesian trust committee election) + AdaptSwitch (PPO protocol switching).

## Layout

- `adatrust/protocols.py` — PBFT/HotStuff/Tendermint performance models (all-BFT pool), action grids
- `adatrust/trust.py` — Beta trust layer, committee sizing (Eq. 9), election, DR/FPR
- `adatrust/adversary.py` — attack classes A1–A4, static/adaptive schedules
- `adatrust/env.py` — epoch-level consensus environment (reward Eq. 6, hysteresis Eq. 11)
- `adatrust/ppo.py` — compact PPO, factorized categorical heads
- `adatrust/policies.py` — baselines B1–B4, ablations, hysteresis gate
- `experiments/download_trace.py` — real Ethereum arrival trace via public JSON-RPC
- `experiments/train.py` — train AdaptSwitch (and B3 tuner, A4 no-switch-cost variant)
- `experiments/run_all.py` — RQ1–RQ5 evaluation, CSV export to `../results/raw/`
- `experiments/plot_all.py` — figures (300 DPI PNG+PDF) to `../results/figures/`

## Reproduce (tested on Ubuntu 22.04, Python 3.10, 2×H100)

```bash
pip install -r requirements.txt
cd experiments
python download_trace.py                       # or skip -> synthetic workloads only
python train.py --out runs/ppo.pt              # main agent (~51k epochs)
python train.py --restrict-pid 1 --out runs/ppo_pbft.pt    # B3 tuner
python train.py --out runs/ppo_nosw.pt ...     # see run log; w4=0 variant
python run_all.py                              # CSVs -> ../results/raw/
python plot_all.py                             # figures -> ../results/figures/
```

Random seeds are fixed ({0,1,2}); all reported numbers are mean ± std over seeds.
