"""Compose the ATL Stomp before/after video (English narration).

Left  BEFORE = stock SONIC, NVIDIA's released base checkpoint, run on the
               choreography at its original tempo: it loses balance at 2.9 s and
               lies on the floor for the rest of the clip.
Right AFTER  = the fine-tuned policy on the same reference, completing every
               control step upright.

Both are rolled out on the *same* motion file, so the comparison is
apples-to-apples.

Both panels are real Isaac Sim RTX renders of a free-dynamics rollout of the
same reference motion (444 control steps @ 50 Hz, every termination except
time_out disabled, so no episode reset can hide a fall). The plot tracks each
policy's root height against the reference band; the numbers panel is computed
from the track files rather than hardcoded.

Usage:
  python make_before_after_video.py --before a.mp4 --after b.mp4 \
      --before-tracks a.npz --after-tracks b.npz --out out.mp4
"""
import argparse

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_W, OUT_H = 1920, 1080
PANEL_W, PANEL_H = 880, 495
PANEL_Y = 92
PANEL_X = [60, 980]
FPS = 50.0

TITLES = [
    ("BEFORE - stock SONIC (base model, no fine-tuning)", "#c9c9c9"),
    ("AFTER - fine-tuned policy (completes the move)", "#5fd35f"),
]
COLOURS = ["#c9c9c9", "#5fd35f"]

# (start_fraction, end_fraction, caption) - fractions so the timings follow the
# clip length instead of being pinned to one motion's duration
CAPTIONS = [
    (0.00, 0.24, "Both policies are given the same reference: the ATL Stomp at its original tempo."),
    (0.24, 0.52, "BEFORE: stock SONIC keeps balance for the first two seconds, then loses the move."),
    (0.52, 0.78, "AFTER: the same moment passes, and it keeps dancing."),
    (0.78, 1.01, "AFTER finishes every control step upright - root height inside the reference band the whole way."),
]


def read_video(path, n):
    cap = cv2.VideoCapture(path)
    out = []
    for _ in range(n):
        ok, img = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(
            cv2.resize(img, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(out)


def root_and_error(path, n):
    d = np.load(path)
    P = d["pred_pos_all"][:n]
    G = d["gt_pos_all"][:n]
    return P[:, 0, 2], np.linalg.norm(P - G, axis=-1).mean(axis=1) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--before-tracks", required=True)
    ap.add_argument("--after-tracks", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    n = 2000  # upper bound; read_video stops at the real end of each clip
    frames = [read_video(a.before, n), read_video(a.after, n)]
    n = min(len(frames[0]), len(frames[1]))
    frames = [f[:n] for f in frames]
    data = [root_and_error(a.before_tracks, n), root_and_error(a.after_tracks, n)]
    zs = [d[0] for d in data]
    errs = [d[1] for d in data]
    gz = np.load(a.after_tracks)["gt_pos_all"][:n, 0, 2]  # same reference for both
    t = np.arange(n) / FPS

    fig = plt.figure(figsize=(OUT_W / 100, OUT_H / 100), dpi=100, facecolor="black")
    fig.text(0.5, 0.972, "G1 learning ATL Stomp  -  base model vs fine-tuned policy",
             color="white", ha="center", va="center", fontsize=21)

    axes = []
    for x, (title, colour) in zip(PANEL_X, TITLES):
        fig.text((x + 8) / OUT_W, 0.928, title, color=colour,
                 ha="left", va="center", fontsize=15, weight="bold")
        ax = fig.add_axes([x / OUT_W, 1 - (PANEL_Y + PANEL_H) / OUT_H,
                           PANEL_W / OUT_W, PANEL_H / OUT_H])
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#333333")
        axes.append(ax)
    ims = [ax.imshow(frames[i][0], animated=True) for i, ax in enumerate(axes)]

    cap_text = fig.text(0.5, 0.442, "", color="white", ha="center", va="center", fontsize=15)
    live = fig.text(0.5, 0.405, "", color="#cccccc", ha="center", va="center", fontsize=13)

    # ---- root-height plot ----
    axp = fig.add_axes([0.055, 0.085, 0.545, 0.29], facecolor="#0d0d0d")
    axp.set_xlim(0, t[-1])
    axp.set_ylim(0.0, 1.0)
    axp.set_xlabel("time (s)", color="#999999", fontsize=11)
    axp.set_ylabel("root height (m)", color="#999999", fontsize=11)
    axp.tick_params(colors="#999999", labelsize=10)
    for s in axp.spines.values():
        s.set_color("#333333")
    axp.axhspan(gz.min(), gz.max(), color="#2b4a2b", alpha=0.55, lw=0)
    axp.text(0.12, gz.max() + 0.02, "reference root height", color="#7fbf7f", fontsize=10)
    axp.axhline(0.45, color="#cc4444", ls="--", lw=1.0)
    axp.text(0.12, 0.39, "fall threshold", color="#cc4444", fontsize=10)
    for z, c, lbl in zip(zs, COLOURS, ("BEFORE root height", "AFTER root height")):
        axp.plot(t, z, color=c, lw=1.8, label=lbl)
    axp.legend(loc="upper right", fontsize=9.5, facecolor="#111111",
               edgecolor="#333333", labelcolor="white")
    cursors = [axp.plot([], [], color="white", lw=1.0, alpha=0.6)[0] for _ in range(2)]

    # ---- numbers panel, computed from the tracks ----
    box = fig.add_axes([0.635, 0.085, 0.32, 0.29], facecolor="#0d0d0d")
    box.set_xticks([])
    box.set_yticks([])
    for s in box.spines.values():
        s.set_color("#333333")
    box.text(0.04, 0.94, "free-dynamics rollout, whole clip", color="#999999",
             fontsize=11, va="top", transform=box.transAxes)
    for x, lbl in zip((0.60, 0.82), ("BEFORE", "AFTER")):
        box.text(x, 0.79, lbl, color="#bbbbbb", fontsize=12, va="top",
                 ha="left", transform=box.transAxes)

    rows = [
        ("fall onset", ["none" if z.min() > 0.45 else
                        f"{int(np.where(z < 0.45)[0][0]) / 50.0:.2f} s" for z in zs]),
        ("root height, minimum", [f"{z.min():.3f} m" for z in zs]),
        ("mean tracking error", [f"{e.mean():.0f} mm" for e in errs]),
        ("worst tracking error", [f"{e.max():.0f} mm" for e in errs]),
    ]
    for i, (label, vals) in enumerate(rows):
        y = 0.65 - i * 0.145
        box.text(0.04, y, label, color="#dddddd", fontsize=11.5, va="top",
                 transform=box.transAxes)
        for x, v, c in zip((0.60, 0.82), vals, COLOURS):
            box.text(x, y, v, color=c, fontsize=11.5, va="top", ha="left",
                     transform=box.transAxes)

    fig.text(0.5, 0.035,
             "Yellow markers = reference targets (closer to the body = better tracking).  "
             "Both rollouts run in free dynamics with every termination except time-out disabled.",
             color="#888888", ha="center", va="center", fontsize=11)

    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (OUT_W, OUT_H))
    assert vw.isOpened(), "VideoWriter failed"

    fig.canvas.draw()
    for i in range(n):
        for k in range(2):
            ims[k].set_data(frames[k][i])
        frac = i / max(n - 1, 1)
        for s, e, txt in CAPTIONS:
            if s <= frac < e:
                cap_text.set_text(txt)
                break
        else:
            cap_text.set_text(CAPTIONS[-1][2])
        for k in range(2):
            cursors[k].set_data(t[:i + 1], zs[k][:i + 1])
        live.set_text(
            f"t = {t[i]:4.2f} s    root height   BEFORE {zs[0][i]:.3f} m   "
            f"AFTER {zs[1][i]:.3f} m")
        fig.canvas.draw()
        vw.write(cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[:, :, :3],
                              cv2.COLOR_RGB2BGR))
        if i % 50 == 0:
            print(f"  frame {i}/{n}", flush=True)

    vw.release()
    print(f"SAVED {a.out}")
    for k, z in zip(("BEFORE", "AFTER"), zs):
        print(f"  {k:7s} root height min {z.min():.3f} m")


if __name__ == "__main__":
    main()
