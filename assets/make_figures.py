"""Generates the animated ACA loop used in the README: assets/aca_loop_{light,dark}.gif.

The animation is illustrative: the step trace below is hand-written, but every rule it
shows (the seven strategies, the forgetting check, and the rollback) matches
aca_channel_estimation.py.

    python assets/make_figures.py            # writes next to this file
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent)
OUT.mkdir(parents=True, exist_ok=True)

THEMES = {
    "light": dict(bg="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#8a8984", grid="#e4e3df",
                  track="#f0efec", accept="#1c5cab", reject="#eb6834", soft="#cde2fb"),
    "dark": dict(bg="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#8f8e86", grid="#383835",
                 track="#2a2a28", accept="#3987e5", reject="#d95926", soft="#184f95"),
}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

# name, trainable part, learning-rate scale, anchor replay  (see ACATrainer.apply_adaptation_action)
STRATEGIES = [
    ("FULL_UPDATE", "all layers", "1x", False),
    ("LAST_LAYER", "output layer", "1x", False),
    ("FREEZE_EARLY", "last 1/2", "1x", False),
    ("SMALL_LR", "all layers", "0.1x", False),
    ("ANCHOR_REPLAY", "all layers", "0.5x", True),
    ("ADAPTIVE_MIX", "last 2/3", "0.5x", True),
    ("CONSERVATIVE", "all layers", "0.05x", False),
]
STAGES = ["Observe state", "Agent selects", "Update", "Check anchor", "Accept / rollback"]

# Illustrative trace: (SNR regime, action index, forgetting, threshold)
TRACE = [
    # (starts at the second regime: the anchor memory is empty while training on the first)
    (10, 0, 0.02, 0.15), (10, 0, 0.04, 0.15), (10, 3, 0.01, 0.15),
    (12, 0, 0.21, 0.15), (12, 4, 0.05, 0.15), (12, 1, 0.03, 0.15),
    (14, 0, 0.06, 0.08), (14, 5, 0.02, 0.08), (14, 0, 0.12, 0.08),
    (16, 6, 0.01, 0.08), (16, 2, 0.04, 0.08), (16, 0, 0.03, 0.08),
]
FMAX = 0.25


def draw_frame(mode, step, stage):
    t = THEMES[mode]
    snr, action, forgetting, threshold = TRACE[step]
    rollback = forgetting > threshold

    fig = plt.figure(figsize=(9.6, 5.6), dpi=100)
    fig.patch.set_facecolor(t["bg"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    ax.text(3, 56.5, "Agentic Continual Adaptation: one decision per mini-batch", fontsize=15,
            fontweight="bold", color=t["ink"], va="center")
    ax.text(3, 52.8, f"SNR regime: {snr} dB   ·   step {step + 1} of {len(TRACE)}   ·   illustrative trace",
            fontsize=10.5, color=t["ink2"], va="center")

    # Stage pipeline
    x0, w, gap = 3, 17.2, 1.8
    for i, name in enumerate(STAGES):
        x = x0 + i * (w + gap)
        active = i == stage
        done = i < stage
        face = t["accept"] if active else (t["soft"] if done else t["track"])
        if active and i == 4 and rollback:
            face = t["reject"]
        ax.add_patch(FancyBboxPatch((x, 44), w, 5, boxstyle="round,pad=0,rounding_size=1.2",
                                    facecolor=face, edgecolor="none"))
        ax.text(x + w / 2, 46.5, name, ha="center", va="center", fontsize=10,
                color="#ffffff" if active else t["ink2"], fontweight="bold" if active else "normal")

    # Strategy list (agent's action space)
    ax.text(3, 40, "Strategy (action)", fontsize=10, color=t["muted"], va="center")
    ax.text(24, 40, "trainable", fontsize=10, color=t["muted"], va="center")
    ax.text(36, 40, "LR", fontsize=10, color=t["muted"], va="center")
    ax.text(42, 40, "replay", fontsize=10, color=t["muted"], va="center")
    for i, (name, part, lr, replay) in enumerate(STRATEGIES):
        y = 36 - i * 4.6
        chosen = stage >= 1 and i == action
        if chosen:
            ax.add_patch(FancyBboxPatch((2, y - 1.9), 46, 3.8, boxstyle="round,pad=0,rounding_size=1",
                                        facecolor=t["soft"], edgecolor=t["accept"], linewidth=1.8))
        fade = 1.0 if (stage < 1 or chosen) else 0.45
        weight = "bold" if chosen else "normal"
        ax.text(3.5, y, name, fontsize=10.5, color=t["ink"], va="center", alpha=fade, fontweight=weight)
        ax.text(24, y, part, fontsize=10, color=t["ink2"], va="center", alpha=fade)
        ax.text(36, y, lr, fontsize=10, color=t["ink2"], va="center", alpha=fade)
        ax.text(44, y, "✓" if replay else "·", fontsize=11, color=t["ink2"], va="center",
                ha="center", alpha=fade)

    # Forgetting check
    ax.text(54, 40, "Forgetting Δ: rise in anchor-memory MSE", fontsize=10,
            color=t["muted"], va="center")
    bx, by, bw, bh = 54, 30, 42, 4
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0,rounding_size=1",
                                facecolor=t["track"], edgecolor="none"))
    if stage >= 3:
        fill = bw * min(forgetting, FMAX) / FMAX
        ax.add_patch(FancyBboxPatch((bx, by), fill, bh, boxstyle="round,pad=0,rounding_size=1",
                                    facecolor=t["reject"] if rollback else t["accept"], edgecolor="none"))
        ax.text(bx + fill + 1, by + bh / 2, f"Δ = {forgetting:.2f}", fontsize=10,
                color=t["ink"], va="center")
    tx = bx + bw * threshold / FMAX
    ax.plot([tx, tx], [by - 1.2, by + bh + 1.2], color=t["ink"], linewidth=1.6)
    ax.text(tx, by - 2.8, f"threshold τ = {threshold:.2f}", fontsize=9.5, color=t["ink2"], ha="center",
            va="center")

    if stage == 4:
        msg = "Δ > τ  →  rollback to checkpoint" if rollback else "Δ ≤ τ  →  keep the update"
        ax.text(bx, 21.5, msg, fontsize=12, fontweight="bold", va="center",
                color=t["reject"] if rollback else t["accept"])

    # Step history
    ax.text(54, 15, "History", fontsize=10, color=t["muted"], va="center")
    for j in range(len(TRACE)):
        x = 55 + j * 3.4
        if j < step or (j == step and stage == 4):
            rb = TRACE[j][2] > TRACE[j][3]
            if rb:
                ax.plot(x, 10, marker="X", markersize=11, color=t["reject"], markeredgecolor=t["bg"])
            else:
                ax.plot(x, 10, marker="o", markersize=10, color=t["accept"], markeredgecolor=t["bg"])
        else:
            ax.plot(x, 10, marker="o", markersize=10, color=t["track"], markeredgecolor=t["bg"])
    ax.plot(55, 4.5, marker="o", markersize=8, color=t["accept"])
    ax.text(56.5, 4.5, "update kept", fontsize=9.5, color=t["ink2"], va="center")
    ax.plot(70, 4.5, marker="X", markersize=9, color=t["reject"])
    ax.text(71.5, 4.5, "rolled back", fontsize=9.5, color=t["ink2"], va="center")

    fig.canvas.draw()
    img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
    plt.close(fig)
    return img


def make_gif(mode):
    frames, durations = [], []
    for step in range(len(TRACE)):
        for stage in range(len(STAGES)):
            frames.append(draw_frame(mode, step, stage))
            durations.append(1100 if stage == 4 else 450)
    durations[-1] = 2500
    palette_frames = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
    palette_frames[0].save(OUT / f"aca_loop_{mode}.gif", save_all=True, append_images=palette_frames[1:],
                           duration=durations, loop=0, optimize=True)


if __name__ == "__main__":
    for mode in THEMES:
        make_gif(mode)
        print(f"wrote {OUT / f'aca_loop_{mode}.gif'}")
