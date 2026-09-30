"""Draw Fig. 1 (AdaTrust architecture) and Fig. 2 (epoch flow / handover
timing) in a submission-grade style for SCI Q1 venues (IEEE TIFS/TDSC/IoT-J):
layered containers, a visually separated fast/slow loop, data-flow vs
control-flow arrows, restrained palette, vector 300-DPI output.

Output: results/figures/fig1_architecture.{png,pdf}, fig2_epoch_flow.{png,pdf}
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

# restrained SCI-Q1 palette
FAST_FC, FAST_EC = "#FBEEE6", "#B4542A"   # warm, fast loop
SLOW_FC, SLOW_EC = "#E9EDF7", "#2C4A7C"   # cool, slow loop
INF_FC, INF_EC = "#F2F2F2", "#555555"     # infrastructure, neutral
ACC_FC, ACC_EC = "#FBE4E4", "#8E2F2F"     # handover (safety-critical)
TXT = "#1a1a1a"


def box(ax, xy, w, h, title, body, fc, ec, tfs=9.5, bfs=7.6, lw=1.4,
        title_weight="bold"):
    # defensive: literal "\n" (e.g. from r-strings) must become a real newline
    if body:
        body = body.replace("\\n", "\n")
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.015,"
                                "rounding_size=0.06",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    cx = xy[0] + w / 2
    ax.text(cx, xy[1] + h - 0.16, title, ha="center", va="center",
            fontsize=tfs, fontweight=title_weight, color=ec, zorder=3)
    if body:
        ax.text(cx, xy[1] + h / 2 - 0.10, body, ha="center", va="center",
                fontsize=bfs, color=TXT, zorder=3, linespacing=1.35)


def container(ax, xy, w, h, label, color, ls="--"):
    ax.add_patch(Rectangle(xy, w, h, fill=False, ec=color, lw=1.1,
                           ls=ls, zorder=1))
    ax.text(xy[0] + 0.12, xy[1] + h - 0.10, label, ha="left", va="center",
            fontsize=8.5, style="italic", color=color, zorder=3)


def arrow(ax, p1, p2, label="", style="-|>", color="#333333", fs=7.6, ls="-",
          dx=0.0, dy=0.0, lw=1.3):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=13,
                                 color=color, lw=lw, linestyle=ls, zorder=4,
                                 shrinkA=1, shrinkB=1))
    if label:
        mx, my = (p1[0] + p2[0]) / 2 + dx, (p1[1] + p2[1]) / 2 + dy
        ax.text(mx, my, label, ha="center", va="center", fontsize=fs,
                color=color, zorder=5,
                bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))


# ================= Fig. 1 — dual-loop architecture =================
fig, ax = plt.subplots(figsize=(11, 5.8))
ax.set_xlim(0, 11); ax.set_ylim(0.9, 5.8); ax.axis("off")

# containers: fast loop (left), slow loop (right); wide central gutter
container(ax, (0.15, 0.95), 4.7, 4.65, "Fast loop (evidence per block, "
          "election per epoch)", FAST_EC)
container(ax, (6.15, 0.95), 4.7, 4.65, "Slow loop (per epoch)", SLOW_EC)

# --- fast loop column ---
box(ax, (0.45, 4.45), 4.1, 0.85, "BayesElect — committee election",
    r"Beta posterior $T_i$, decay $\lambda_d$, slash $\kappa$"
    "\n" r"committee sizing $k^*$ (Eq. 5)", FAST_FC, FAST_EC)
box(ax, (0.45, 2.90), 4.1, 0.85, "Evidence & logging plane",
    "votes / slashes / decisions (append-only)", INF_FC, INF_EC)
box(ax, (0.45, 1.15), 4.1, 0.80, "Blockchain network",
    r"$n$ validators, partial synchrony, load $\lambda(t)$", INF_FC, INF_EC)

# --- slow loop column ---
box(ax, (6.45, 4.45), 4.1, 0.85, "AdaptSwitch — PPO controller",
    r"state $s_t$, joint action $(p,B,\tau,k)$"
    "\n" r"hysteresis $(\delta_h, d_{\min})$", SLOW_FC, SLOW_EC)
box(ax, (6.45, 2.90), 4.1, 0.85, "Handover manager",
    "bounded-overlap invariant (Thm. 2)", ACC_FC, ACC_EC)
box(ax, (6.45, 1.15), 4.1, 0.80, "Protocol pool",
    "PBFT  |  HotStuff  |  Tendermint", INF_FC, INF_EC)

# --- fast-loop internal arrows (data flow; arrows live in the gaps) ---
arrow(ax, (2.5, 1.95), (2.5, 2.90), "on-chain evidence", color="#777777")
arrow(ax, (2.5, 3.75), (2.5, 4.45), "trust update / $k^*$", color="#777777")

# --- slow-loop internal arrows (control flow) ---
arrow(ax, (7.5, 4.45), (7.5, 3.75), r"action $(p,B,\tau,k)$", color=SLOW_EC)
arrow(ax, (7.5, 2.90), (7.5, 1.95), "activate $p_e$", color=SLOW_EC)

# --- cross-loop / cross-layer (labels sit in the wide central gutter) ---
arrow(ax, (4.55, 4.85), (6.45, 4.85), r"trust summary $\mathbf{u}_t$",
      color="#555555")
arrow(ax, (4.55, 3.30), (6.45, 3.30), r"elected committee $\mathcal{C}_e$",
      color="#555555")
arrow(ax, (6.45, 1.55), (4.55, 1.55), "run consensus ($E$ blocks)",
      color="#777777", ls="--")

fig.savefig(FIG / "fig1_architecture.png")
fig.savefig(FIG / "fig1_architecture.pdf")
plt.close(fig)

# ================= Fig. 2 — epoch flow =================
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 3.2); ax.axis("off")

phases = [
    ("1 · Observe", "finalize epoch log\ntrust update (Alg. 1)", INF_FC, INF_EC),
    ("2 · Decide", "PPO + hysteresis\n(Eq. 6)", SLOW_FC, SLOW_EC),
    ("3 · Elect", "$k^*$ sizing + sampling\n(Eq. 5)", FAST_FC, FAST_EC),
    ("4 · Handover", "drain to $h^*$,\nactivate $h^*{+}1$\n(Thm. 2)", ACC_FC, ACC_EC),
    ("5 · Execute", "$E$ blocks under $p_e$\nevidence → log", INF_FC, INF_EC),
]
x0, w, gap = 0.25, 1.74, 0.22
for i, (title, body, fc, ec) in enumerate(phases):
    xi = x0 + i * (w + gap)
    box(ax, (xi, 1.35), w, 1.15, title, body, fc, ec)
    if i:
        arrow(ax, (x0 + i * (w + gap) - gap, 1.93),
              (x0 + i * (w + gap), 1.93), color="#555555")
# next-epoch return loop (below the strip)
xr = x0 + 4 * (w + gap) + w
arrow(ax, (xr, 1.35), (xr, 0.55), color="#777777", ls="--")
arrow(ax, (xr, 0.55), (x0 + w / 2, 0.55), "next epoch", color="#777777",
      ls="--")
arrow(ax, (x0 + w / 2, 0.55), (x0 + w / 2, 1.35), color="#777777", ls="--")

fig.savefig(FIG / "fig2_epoch_flow.png")
fig.savefig(FIG / "fig2_epoch_flow.pdf")
plt.close(fig)

print("[fig] fig1_architecture, fig2_epoch_flow ->", FIG)
