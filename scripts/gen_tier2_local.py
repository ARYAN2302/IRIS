#!/usr/bin/env python3
"""
gen_tier2_local.py — Tier 2 visuals: Architecture diagram + Radar chart
Run locally on your Mac (no Modal needed).

    cd /Users/adarshthakur/Desktop/IRIS/scripts
    python gen_tier2_local.py

Output -> /Users/adarshthakur/Desktop/IRIS/iris_output/
    05_architecture_diagram.png
    06_radar_chart.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

OUTPUT_DIR = '/Users/adarshthakur/Desktop/IRIS/iris_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Dark theme ───
BG = '#0d1117'
FG = '#e6edf3'
BORDER = '#30363d'
MUTED = '#8b949e'

# Real per-type AUC from v11 holdout eval
PER_TYPE_AUC = {
    'RadioMaster BOXER':  1.0000,
    'WFLY ET10':          1.0000,
    'JUMPER-T14':         0.9999,
    'JR PROPO XG7':       0.9997,
    'FUTABA-T10J':        0.9984,
    'DJI FPV COMBO':      0.9674,
    'FUTABA-T14SG':       0.9656,
}


# ═══════════════════════════════════════════════════════════════
# #5: LeJEPA ARCHITECTURE DIAGRAM (v11: Hierarchical SupCon)
# ═══════════════════════════════════════════════════════════════
def gen_architecture_diagram():
    fig, ax = plt.subplots(figsize=(18, 10))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Color palette
    C_INPUT     = '#58a6ff'
    C_ENCODER   = '#f0883e'
    C_PROJECTOR = '#bc8cff'
    C_PREDICTOR = '#3fb950'
    C_LOSS      = '#f85149'
    C_SIGREG    = '#79c0ff'
    C_FINE      = '#d2a8ff'
    C_COARSE    = '#ffa657'
    C_HIER      = '#a5d6ff'
    C_STOPGRAD  = '#8b949e'
    C_BOX_BG    = '#161b22'

    def draw_box(x, y, w, h, label, color, sublabel=''):
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.12",
            facecolor=color, edgecolor=BORDER,
            linewidth=2, zorder=2, alpha=0.92
        )
        ax.add_patch(rect)
        if sublabel:
            ax.text(x + w/2, y + h/2 + 0.15, label,
                    ha='center', va='center', fontsize=11,
                    fontweight='bold', color=BG, zorder=3)
            ax.text(x + w/2, y + h/2 - 0.18, sublabel,
                    ha='center', va='center', fontsize=8,
                    color=BG, zorder=3, alpha=0.75)
        else:
            ax.text(x + w/2, y + h/2, label,
                    ha='center', va='center', fontsize=11,
                    fontweight='bold', color=BG, zorder=3)

    def draw_arrow(x1, y1, x2, y2, color=MUTED, lw=2, style='->'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw))

    def draw_dashed_arrow(x1, y1, x2, y2, color=MUTED, lw=1.5):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                    linestyle='dashed'))

    # ── Title ──
    ax.text(9, 9.4, 'LeJEPA + SIGReg + Hierarchical SupCon', ha='center', va='center',
            fontsize=22, fontweight='bold', color=FG)
    ax.text(9, 8.9, 'Self-supervised drone-ness learning from RF spectrograms (v11)',
            ha='center', va='center', fontsize=12, color=MUTED)

    # ── ROW 1: Drone path (y=6.5) ──
    draw_box(0.5,  6.5, 2.3, 0.9, 'x\u2081 (Drone)', C_INPUT)
    draw_box(3.3,  6.5, 2.3, 0.9, 'Encoder f\u03b8', C_ENCODER, '6-block CNN+BN')
    draw_box(6.1,  6.5, 2.3, 0.9, 'Projector g\u03b8', C_PROJECTOR, 'MLP 256\u2192256')
    draw_box(8.9,  6.5, 2.3, 0.9, 'Predictor h\u03b8', C_PREDICTOR, 'MLP 256\u2192256')

    # ── ROW 2: Matched BG path (y=3.8) ──
    draw_box(0.5,  3.8, 2.3, 0.9, 'x\u2082 (Matched BG)', C_INPUT)
    draw_box(3.3,  3.8, 2.3, 0.9, 'Encoder f\u03b8', C_ENCODER, '6-block CNN+BN')
    draw_box(6.1,  3.8, 2.3, 0.9, 'Projector g\u03b8', C_PROJECTOR, 'MLP 256\u2192256')
    draw_box(8.9,  3.8, 2.3, 0.9, 'sg(z\u2082)', C_STOPGRAD, 'stop-gradient')

    # ── Loss boxes ──
    draw_box(12.5, 5.8, 2.8, 0.8, 'L_align', C_LOSS, '||h(z\u2081) - sg(z\u2082)||\u00b2')

    # SIGReg box
    draw_box(2.5, 1.3, 3.0, 0.8, 'L_SIGReg', C_SIGREG, '\u03c3(z)\u21921, \u03bc(z)\u21920')
    ax.text(4.0, 0.9, '\u03bb = 0.001', ha='center', fontsize=9, color=C_SIGREG)

    # ── Hierarchical SupCon decomposition (the key v11 change) ──
    # Fine-grained loss
    draw_box(6.5, 1.3, 3.0, 0.8, 'L_fine', C_FINE, 'same type = positive')
    ax.text(8.0, 0.9, 'w = 0.7', ha='center', fontsize=9, color=C_FINE)

    # Coarse-grained loss
    draw_box(10.5, 1.3, 3.0, 0.8, 'L_coarse', C_COARSE, 'all drones = positive')
    ax.text(12.0, 0.9, 'w = 0.3', ha='center', fontsize=9, color=C_COARSE)

    # Hierarchical SupCon combined
    draw_box(8.5, 0.0, 4.0, 0.7, 'L_HierSupCon = 0.7\u00b7L_fine + 0.3\u00b7L_coarse',
             C_HIER, '')

    # Total loss
    draw_box(3.0, -1.2, 11.5, 0.8, 'L = (1\u2212\u03bb)L_align + \u03bbL_SIGReg + \u03b1L_HierSupCon',
             '#238636', '')
    ax.text(8.75, -1.7, '\u03bb=0.001, \u03b1=0.05, T=0.07', ha='center', fontsize=9, color=MUTED)

    # ── Arrows: Drone path ──
    draw_arrow(2.8, 6.95, 3.3, 6.95, C_INPUT)
    draw_arrow(5.6, 6.95, 6.1, 6.95, C_ENCODER)
    draw_arrow(8.4, 6.95, 8.9, 6.95, C_PROJECTOR)
    draw_arrow(11.2, 6.95, 12.5, 6.2, C_PREDICTOR)

    # ── Arrows: BG path ──
    draw_arrow(2.8, 4.25, 3.3, 4.25, C_INPUT)
    draw_arrow(5.6, 4.25, 6.1, 4.25, C_ENCODER)
    draw_arrow(8.4, 4.25, 8.9, 4.25, C_PROJECTOR)
    draw_arrow(11.2, 4.25, 12.5, 5.8, C_STOPGRAD)

    # ── SIGReg arrows ──
    draw_dashed_arrow(7.25, 3.8, 4.5, 2.1, C_SIGREG)
    draw_dashed_arrow(7.25, 6.5, 4.5, 2.1, C_SIGREG)

    # ── Fine loss arrows ──
    draw_dashed_arrow(7.25, 6.5, 8.0, 2.1, C_FINE)
    draw_dashed_arrow(7.25, 3.8, 8.0, 2.1, C_FINE)

    # ── Coarse loss arrows ──
    draw_dashed_arrow(7.25, 6.5, 12.0, 2.1, C_COARSE)
    draw_dashed_arrow(7.25, 3.8, 12.0, 2.1, C_COARSE)

    # ── Fine/Coarse -> HierSupCon arrows ──
    draw_arrow(8.0, 1.3, 10.0, 0.7, C_FINE, lw=1.5)
    draw_arrow(12.0, 1.3, 11.0, 0.7, C_COARSE, lw=1.5)

    # ── Shared-weights indicator ──
    ax.text(4.45, 5.7, 'shared', ha='center', va='center',
            fontsize=9, color=C_ENCODER, fontstyle='italic')
    draw_arrow(4.45, 6.5, 4.45, 6.0, C_ENCODER, lw=1, style='->')
    draw_arrow(4.45, 4.7, 4.45, 5.2, C_ENCODER, lw=1, style='->')

    ax.text(7.25, 5.7, 'shared', ha='center', va='center',
            fontsize=9, color=C_PROJECTOR, fontstyle='italic')
    draw_arrow(7.25, 6.5, 7.25, 6.0, C_PROJECTOR, lw=1, style='->')
    draw_arrow(7.25, 4.7, 7.25, 5.2, C_PROJECTOR, lw=1, style='->')

    # ── Stop-gradient label ──
    ax.text(10.05, 3.4, 'stop-grad', ha='center', va='center',
            fontsize=9, color=MUTED, fontstyle='italic')

    # ── Key insight box ──
    insight_box = FancyBboxPatch(
        (14.0, 0.5), 3.5, 3.5,
        boxstyle="round,pad=0.2",
        facecolor=C_BOX_BG, edgecolor=C_COARSE,
        linewidth=2, zorder=2, alpha=0.9
    )
    ax.add_patch(insight_box)
    ax.text(15.75, 3.6, 'v11 KEY CHANGE', ha='center', fontsize=10,
            fontweight='bold', color=C_COARSE)
    ax.text(15.75, 3.1, 'Hierarchical labels:', ha='center', fontsize=9,
            fontweight='bold', color=FG)
    ax.text(15.75, 2.6, 'Parent: drone / background', ha='center', fontsize=8.5, color=MUTED)
    ax.text(15.75, 2.2, 'Child: 30 drone types', ha='center', fontsize=8.5, color=MUTED)
    ax.text(15.75, 1.6, 'Coarse loss merges', ha='center', fontsize=9,
            fontweight='bold', color=C_COARSE)
    ax.text(15.75, 1.2, '30 clusters into one', ha='center', fontsize=9,
            fontweight='bold', color=C_COARSE)
    ax.text(15.75, 0.8, '"drone" region', ha='center', fontsize=9,
            fontweight='bold', color=C_COARSE)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, '05_architecture_diagram.png')
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'Saved: {out}')


# ═══════════════════════════════════════════════════════════════
# #6: PER-TYPE RADAR CHART (v11)
# ═══════════════════════════════════════════════════════════════
def gen_radar_chart():
    types = list(PER_TYPE_AUC.keys())
    aucs  = list(PER_TYPE_AUC.values())
    N = len(types)

    # Angles for each spoke
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    # Close the polygon
    aucs_plot = aucs + [aucs[0]]
    angles_plot = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # Style the polar axes
    ax.spines['polar'].set_color(BORDER)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.set_ylim(0.90, 1.02)
    ax.set_yticks([0.92, 0.94, 0.96, 0.98, 1.0])
    ax.set_yticklabels(['0.92', '0.94', '0.96', '0.98', '1.00'],
                       color=MUTED, fontsize=9)
    ax.yaxis.grid(True, color=BORDER, linestyle='--', linewidth=0.5)
    ax.xaxis.grid(True, color=BORDER, linewidth=0.8)

    # Plot the radar polygon
    ax.plot(angles_plot, aucs_plot, color='#58a6ff', linewidth=2.5, zorder=3)
    ax.fill(angles_plot, aucs_plot, color='#58a6ff', alpha=0.15, zorder=2)

    # Plot data points
    ax.scatter(angles, aucs, color='#58a6ff', s=80, zorder=4, edgecolors='white', linewidths=1)

    # Add AUC value labels next to each point
    for angle, auc_val, name in zip(angles, aucs, types):
        offset = 0.025
        ax.text(angle, auc_val + offset, f'{auc_val:.3f}',
                ha='center', va='center', fontsize=10,
                fontweight='bold', color=FG, zorder=5)

    # Set the spoke labels (type names)
    ax.set_xticks(angles)
    ax.set_xticklabels(types, fontsize=10, color=FG, fontweight='bold')

    # Title
    ax.set_title('Per-Type Holdout AUC (v11)', pad=25,
                 fontsize=16, fontweight='bold', color=FG)

    # Add a note about overall
    fig.text(0.5, 0.02,
             'Overall matched-bg AUC: 0.9785  |  7 holdout types  |  7/7 >= 0.95  |  Training: LeJEPA + SIGReg + Hierarchical SupCon',
             ha='center', fontsize=10, color=MUTED)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, '06_radar_chart.png')
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'Saved: {out}')


# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating Tier 2 visuals (v11)...')
    gen_architecture_diagram()
    gen_radar_chart()
    print('Done! Both files in iris_output/')
