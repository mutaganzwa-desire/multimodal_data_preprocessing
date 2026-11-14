"""Draw the system flow diagram used in the README, the report and notebook 4.

    python reports/make_system_flow.py      ->  reports/figures/system_flow.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

OUT = Path(__file__).resolve().parent / "figures" / "system_flow.png"

INK, MUTED = "#1f2a37", "#5b6675"
BLUE, BLUE_EDGE = "#e3eefc", "#2a78d6"
PURPLE, PURPLE_EDGE = "#efe6fb", "#7c4dcc"
GREEN, GREEN_EDGE = "#e2f5ea", "#1b8a50"
RED, RED_EDGE = "#fde8e8", "#c62828"

fig, ax = plt.subplots(figsize=(16, 6.4), dpi=150)
ax.set_xlim(0, 16)
ax.set_ylim(0, 6.4)
ax.axis("off")


def box(x, y, w, h, fill, edge, title, sub=None, radius=0.25):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=f"round,pad=0.02,rounding_size={radius}",
                                fc=fill, ec=edge, lw=2))
    ax.text(x, y + h / 2 - 0.32, title, ha="center", va="center", fontsize=12.5, weight="bold", color=INK)
    if sub:
        ax.text(x, y - 0.12, sub, ha="center", va="center", fontsize=9.5, color=MUTED, linespacing=1.35)


def diamond(x, y, w, h, title, sub):
    ax.add_patch(Polygon([(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)],
                         closed=True, fc=BLUE, ec=BLUE_EDGE, lw=2))
    ax.text(x, y + 0.38, title, ha="center", va="center", fontsize=12, weight="bold", color=INK, linespacing=1.15)
    ax.text(x, y - 0.38, sub, ha="center", va="center", fontsize=9.5, color=MUTED, linespacing=1.3)


def arrow(x1, y1, x2, y2, label=None, colour=INK, label_xy=None):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18, lw=1.8, color=colour))
    if label:
        lx, ly = label_xy or ((x1 + x2) / 2, (y1 + y2) / 2 + 0.22)
        ax.text(lx, ly, label, ha="center", va="center", fontsize=10, weight="bold", color=colour,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"))


Y = 4.2
# Start
ax.add_patch(plt.Circle((1.0, Y), 0.62, fc=PURPLE, ec=PURPLE_EDGE, lw=2))
ax.text(1.0, Y + 0.08, "Start", ha="center", va="center", fontsize=12.5, weight="bold", color=INK)
ax.text(1.0, Y - 0.25, "face photo", ha="center", va="center", fontsize=9, color=MUTED)

diamond(4.0, Y, 3.4, 2.7, "Facial\nRecognition", "confidence \u2265 0.80\nsimilarity \u2265 0.36")
box(7.75, Y, 2.8, 1.75, PURPLE, PURPLE_EDGE, "Product Model",
    "predicts for the member's\ncustomer account\n(result held back)")
diamond(11.4, Y, 3.4, 2.7, "Voice\nValidation", "same member as face\nconfidence \u2265 0.50\nsimilarity \u2265 0.40")
box(14.85, Y, 2.1, 1.75, GREEN, GREEN_EDGE, "Display", "predicted product\n+ ranked categories")

box(4.0, 1.0, 2.5, 0.95, RED, RED_EDGE, "Access Denied", "unknown face / no face")
box(11.4, 1.0, 2.5, 0.95, RED, RED_EDGE, "Access Denied", "voice does not match")

arrow(1.62, Y, 2.35, Y)
arrow(5.7, Y, 6.33, Y, "Pass", GREEN_EDGE)
arrow(9.17, Y, 9.68, Y)
ax.text(9.42, Y - 0.32, "voice\nclip", ha="center", va="top", fontsize=8.5, color=MUTED, style="italic")
arrow(13.1, Y, 13.78, Y, "Pass", GREEN_EDGE)
arrow(4.0, Y - 1.35, 4.0, 1.48, "Fail", RED_EDGE, (4.0, 2.2))
arrow(11.4, Y - 1.35, 11.4, 1.48, "Fail", RED_EDGE, (11.4, 2.2))

ax.text(8.0, 6.05, "User Identity and Product Recommendation System Flow", ha="center", va="center",
        fontsize=16, weight="bold", color=INK)
ax.text(8.0, 0.2, "The prediction is computed after the face check but only shown after the voice of the SAME member "
        "approves it.   Implemented in app.py.", ha="center", va="center", fontsize=9.5, color=MUTED)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print("saved", OUT)
