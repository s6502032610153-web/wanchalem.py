“””
ระบบออกแบบฐานรากรับโมเมนต์และตรวจสอบแรงในเสาเข็ม
กรณีเสาเข็มเยื้องศูนย์ - มาตรฐาน มยผ.1106-64
“””

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, FancyBboxPatch
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import warnings
warnings.filterwarnings(‘ignore’)

# ─────────────────────────────────────────────────────────

# PAGE CONFIG

# ─────────────────────────────────────────────────────────

st.set_page_config(
page_title=“ระบบออกแบบฐานรากเสาเข็ม | มยผ.1106-64”,
page_icon=“🏗️”,
layout=“wide”,
initial_sidebar_state=“expanded”,
)

# ─────────────────────────────────────────────────────────

# CUSTOM CSS

# ─────────────────────────────────────────────────────────

st.markdown(”””

<style>
@import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary: #0a0e1a;
    --bg-secondary: #111827;
    --bg-card: #1a2235;
    --bg-card-hover: #1f2a40;
    --accent-gold: #f5a623;
    --accent-teal: #00d4aa;
    --accent-blue: #4a9eff;
    --accent-red: #ff6b6b;
    --accent-green: #51cf66;
    --text-primary: #e8eaf0;
    --text-secondary: #8892a4;
    --border-subtle: rgba(255,255,255,0.07);
    --border-accent: rgba(245,166,35,0.3);
    --shadow-glow: 0 0 30px rgba(245,166,35,0.1);
}

/* ── Global Reset ── */
.stApp {
    background: var(--bg-primary);
    font-family: 'Sarabun', sans-serif;
}

/* ── Header Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0d1b2a 0%, #1a2942 40%, #0f2035 100%);
    border: 1px solid var(--border-accent);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-glow), inset 0 1px 0 rgba(255,255,255,0.05);
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(245,166,35,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent-gold);
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 1rem;
    color: var(--text-secondary);
    margin: 0;
}
.hero-badge {
    display: inline-block;
    background: rgba(245,166,35,0.15);
    border: 1px solid var(--border-accent);
    color: var(--accent-gold);
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 12px;
    letter-spacing: 0.5px;
}

/* ── Cards ── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    transition: all 0.2s ease;
}
.card:hover {
    border-color: rgba(245,166,35,0.2);
    background: var(--bg-card-hover);
}
.card-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 16px;
    padding-bottom: 10px;
    border-bottom: 1px solid var(--border-subtle);
}
.card-title span {
    color: var(--accent-gold);
    margin-right: 8px;
}

/* ── Result Cards ── */
.result-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin: 16px 0;
}
.result-item {
    background: linear-gradient(135deg, #1a2235, #1f2a40);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    transition: transform 0.2s;
}
.result-item:hover { transform: translateY(-2px); }
.result-label {
    font-size: 0.72rem;
    color: var(--text-secondary);
    margin-bottom: 8px;
    letter-spacing: 0.5px;
}
.result-value {
    font-size: 1.5rem;
    font-weight: 700;
    font-family: 'IBM Plex Mono', monospace;
    color: var(--accent-teal);
}
.result-unit {
    font-size: 0.72rem;
    color: var(--text-secondary);
    margin-top: 2px;
}

/* ── Status Badges ── */
.status-ok {
    background: rgba(81,207,102,0.15);
    border: 1px solid rgba(81,207,102,0.4);
    color: #51cf66;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
    display: inline-block;
}
.status-fail {
    background: rgba(255,107,107,0.15);
    border: 1px solid rgba(255,107,107,0.4);
    color: #ff6b6b;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
    display: inline-block;
}
.status-warn {
    background: rgba(245,166,35,0.15);
    border: 1px solid rgba(245,166,35,0.4);
    color: var(--accent-gold);
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
    display: inline-block;
}

/* ── Formula Box ── */
.formula-box {
    background: #0d1420;
    border-left: 3px solid var(--accent-gold);
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 12px 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: var(--accent-teal);
    line-height: 1.8;
}

/* ── Table ── */
.data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    margin: 12px 0;
}
.data-table th {
    background: rgba(245,166,35,0.1);
    color: var(--accent-gold);
    padding: 10px 14px;
    text-align: center;
    font-weight: 600;
    border-bottom: 2px solid var(--border-accent);
    font-size: 0.78rem;
    letter-spacing: 0.5px;
}
.data-table td {
    padding: 9px 14px;
    text-align: center;
    color: var(--text-primary);
    border-bottom: 1px solid var(--border-subtle);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
}
.data-table tr:hover td { background: rgba(255,255,255,0.03); }
.td-critical { color: var(--accent-red) !important; font-weight: 700; }
.td-ok { color: var(--accent-green) !important; }
.td-warn { color: var(--accent-gold) !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-subtle);
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--accent-gold);
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* ── Streamlit Widgets ── */
.stSlider > div > div > div { background: var(--accent-gold) !important; }
.stNumberInput input, .stTextInput input, .stSelectbox select {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #f5a623, #e8951a) !important;
    color: #0a0e1a !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-family: 'Sarabun', sans-serif !important;
    font-size: 1rem !important;
    padding: 12px 32px !important;
    width: 100% !important;
    letter-spacing: 0.5px;
    transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(245,166,35,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(245,166,35,0.45) !important;
}

/* ── Divider ── */
hr { border-color: var(--border-subtle) !important; margin: 20px 0 !important; }

/* ── Warning / Info box ── */
.stAlert { border-radius: 10px !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary);
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-secondary) !important;
    border-radius: 8px !important;
    font-family: 'Sarabun', sans-serif !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent-gold) !important;
    color: #0a0e1a !important;
    font-weight: 700 !important;
}
</style>

“””, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────

# HELPER: PILE LAYOUT GENERATOR

# ─────────────────────────────────────────────────────────

def generate_pile_layout(pile_config: str, spacing: float, eccentricity_x: float, eccentricity_y: float):
“”“Return pile coordinates (local from group centroid).”””
s = spacing
configs = {
“1 เสาเข็ม”: [(0, 0)],
“2 เสาเข็ม (แนวแกน X)”: [(-s/2, 0), (s/2, 0)],
“2 เสาเข็ม (แนวแกน Y)”: [(0, -s/2), (0, s/2)],
“3 เสาเข็ม (สามเหลี่ยม)”: [(0, 2*s/3), (-s/2, -s/3), (s/2, -s/3)],
“4 เสาเข็ม (สี่เหลี่ยม)”: [(-s/2, -s/2), (s/2, -s/2), (-s/2, s/2), (s/2, s/2)],
“5 เสาเข็ม (X+Center)”: [(-s/2, -s/2), (s/2, -s/2), (-s/2, s/2), (s/2, s/2), (0, 0)],
“6 เสาเข็ม (2x3)”: [(-s, -s/2), (0, -s/2), (s, -s/2), (-s, s/2), (0, s/2), (s, s/2)],
“8 เสาเข็ม (2x4)”: [(-1.5*s, -s/2), (-0.5*s, -s/2), (0.5*s, -s/2), (1.5*s, -s/2),
(-1.5*s,  s/2), (-0.5*s,  s/2), (0.5*s,  s/2), (1.5*s,  s/2)],
“9 เสาเข็ม (3x3)”: [(-s, -s), (0, -s), (s, -s),
(-s,  0), (0,  0), (s,  0),
(-s,  s), (0,  s), (s,  s)],
}
piles = configs.get(pile_config, [(0, 0)])
return [(x + eccentricity_x, y + eccentricity_y) for (x, y) in piles]

# ─────────────────────────────────────────────────────────

# CORE CALCULATION: มยผ.1106-64

# ─────────────────────────────────────────────────────────

def calculate_pile_forces(P, Mx, My, pile_coords):
“””
คำนวณแรงในเสาเข็ม ตามมาตรฐาน มยผ.1106-64
สมการ: Qi = P/n ± Mx·yi/Σyi² ± My·xi/Σxi²
“””
n = len(pile_coords)
xs = np.array([c[0] for c in pile_coords])
ys = np.array([c[1] for c in pile_coords])

```
# จุดศูนย์ถ่วงกลุ่มเสาเข็ม
cx = np.mean(xs)
cy = np.mean(ys)

# ระยะจากจุดศูนย์ถ่วง
xi = xs - cx
yi = ys - cy

sum_xi2 = np.sum(xi**2)
sum_yi2 = np.sum(yi**2)

# แรงในแต่ละเสาเข็ม (ton)
Q_axial = P / n
Q_mx = (Mx * yi / sum_yi2) if sum_yi2 > 1e-9 else np.zeros(n)
Q_my = (My * xi / sum_xi2) if sum_xi2 > 1e-9 else np.zeros(n)

Qi = Q_axial + Q_mx + Q_my

return {
    "n": n,
    "Qi": Qi,
    "Q_axial": Q_axial,
    "Q_mx_contrib": Q_mx,
    "Q_my_contrib": Q_my,
    "xi": xi,
    "yi": yi,
    "cx": cx,
    "cy": cy,
    "sum_xi2": sum_xi2,
    "sum_yi2": sum_yi2,
    "Qmax": np.max(Qi),
    "Qmin": np.min(Qi),
}
```

# ─────────────────────────────────────────────────────────

# VISUALIZATION

# ─────────────────────────────────────────────────────────

def plot_pile_plan(pile_coords, Qi, Qa_comp, Qa_tens, eccentricity_x, eccentricity_y, P, Mx, My):
“”“แผนผังเสาเข็ม พร้อมแสดงแรงในแต่ละต้น”””
fig = plt.figure(figsize=(14, 10), facecolor=’#0a0e1a’)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

```
ax1 = fig.add_subplot(gs[:, 0])   # Plan view (large)
ax2 = fig.add_subplot(gs[0, 1])   # Force bar chart
ax3 = fig.add_subplot(gs[1, 1])   # Utilization ratio

xs = [c[0] for c in pile_coords]
ys = [c[1] for c in pile_coords]
n = len(pile_coords)

# ── Normalize for color ──
norm = Normalize(vmin=min(Qi), vmax=max(Qi))
cmap = cm.get_cmap('RdYlGn_r')

# ── PLAN VIEW ──
ax1.set_facecolor('#0d1420')
ax1.set_aspect('equal')

# Footprint
margin = 0.8
if len(xs) > 1:
    xmin, xmax = min(xs) - margin, max(xs) + margin
    ymin, ymax = min(ys) - margin, max(ys) + margin
else:
    xmin, xmax = -1, 1
    ymin, ymax = -1, 1

foot = FancyBboxPatch((xmin, ymin), xmax-xmin, ymax-ymin,
                       boxstyle="round,pad=0.1",
                       facecolor='#1a2235', edgecolor='#f5a623',
                       linewidth=1.5, linestyle='--', zorder=1)
ax1.add_patch(foot)

# Grid lines
for x in np.arange(np.floor(xmin), np.ceil(xmax)+1, 1.0):
    ax1.axvline(x, color='#ffffff08', linewidth=0.5, zorder=0)
for y in np.arange(np.floor(ymin), np.ceil(ymax)+1, 1.0):
    ax1.axhline(y, color='#ffffff08', linewidth=0.5, zorder=0)

# Pile circles
for i, (x, y) in enumerate(pile_coords):
    color = cmap(norm(Qi[i]))
    ratio = Qi[i] / Qa_comp if Qi[i] >= 0 else abs(Qi[i]) / Qa_tens

    # Outer glow
    glow = Circle((x, y), 0.32, color=color, alpha=0.15, zorder=2)
    ax1.add_patch(glow)
    # Pile body
    pile = Circle((x, y), 0.22, facecolor=color, edgecolor='white',
                  linewidth=1.5, zorder=3)
    ax1.add_patch(pile)
    # Pile number
    ax1.text(x, y, str(i+1), ha='center', va='center',
             fontsize=8, fontweight='bold', color='#0a0e1a', zorder=4)
    # Force label
    label_color = '#ff6b6b' if Qi[i] < 0 else ('#51cf66' if ratio <= 0.8 else '#f5a623')
    ax1.text(x, y - 0.42, f'{Qi[i]:.1f} t', ha='center', va='top',
             fontsize=7.5, color=label_color, fontweight='600', zorder=4,
             fontfamily='monospace')

# Column position (load application point)
col_x = eccentricity_x
col_y = eccentricity_y
col_patch = FancyBboxPatch((col_x-0.18, col_y-0.18), 0.36, 0.36,
                            boxstyle="round,pad=0.02",
                            facecolor='#f5a623', edgecolor='#fff', linewidth=1.5, zorder=5)
ax1.add_patch(col_patch)
ax1.text(col_x, col_y, 'เสา', ha='center', va='center',
         fontsize=6.5, fontweight='bold', color='#0a0e1a', zorder=6)

# Load arrows
arrow_kw = dict(arrowstyle='->', color='#4a9eff', lw=1.5)
ax1.annotate('', xy=(col_x, col_y + 0.6), xytext=(col_x, col_y + 1.1),
             arrowprops=dict(arrowstyle='->', color='#4a9eff', lw=1.8))
ax1.text(col_x + 0.1, col_y + 0.85, f'P={P:.0f}t', fontsize=7,
         color='#4a9eff', fontweight='600', fontfamily='monospace')

if abs(Mx) > 0.01:
    ax1.annotate('', xy=(col_x + 0.5, col_y + 0.15), xytext=(col_x + 0.15, col_y + 0.5),
                 arrowprops=dict(arrowstyle='->', color='#00d4aa', lw=1.5,
                                 connectionstyle='arc3,rad=0.4'))
    ax1.text(col_x + 0.65, col_y + 0.3, f'Mx={Mx:.0f}', fontsize=6.5,
             color='#00d4aa', fontfamily='monospace')

if abs(My) > 0.01:
    ax1.annotate('', xy=(col_x + 0.15, col_y - 0.5), xytext=(col_x + 0.5, col_y - 0.15),
                 arrowprops=dict(arrowstyle='->', color='#ff9f43', lw=1.5,
                                 connectionstyle='arc3,rad=0.4'))
    ax1.text(col_x + 0.55, col_y - 0.45, f'My={My:.0f}', fontsize=6.5,
             color='#ff9f43', fontfamily='monospace')

# Centroid marker
cx_g = np.mean(xs)
cy_g = np.mean(ys)
ax1.plot(cx_g, cy_g, '+', color='#f5a623', markersize=12, markeredgewidth=2, zorder=6)
ax1.text(cx_g + 0.15, cy_g + 0.08, 'CG', fontsize=6.5, color='#f5a623',
         fontweight='600')

# Axes arrows
ax1.annotate('', xy=(xmax-0.1, ymin+0.3), xytext=(xmax-0.6, ymin+0.3),
             arrowprops=dict(arrowstyle='->', color='#8892a4', lw=1))
ax1.text(xmax-0.05, ymin+0.3, 'X', fontsize=7.5, color='#8892a4', va='center')
ax1.annotate('', xy=(xmin+0.3, ymax-0.1), xytext=(xmin+0.3, ymax-0.6),
             arrowprops=dict(arrowstyle='->', color='#8892a4', lw=1))
ax1.text(xmin+0.3, ymax-0.02, 'Y', fontsize=7.5, color='#8892a4', ha='center')

ax1.set_xlim(xmin - 0.5, xmax + 0.5)
ax1.set_ylim(ymin - 0.5, ymax + 0.9)
ax1.set_title('แผนผังกลุ่มเสาเข็ม (Plan View)', color='#f5a623',
              fontsize=11, fontweight='700', pad=12,
              fontfamily='Sarabun')
ax1.set_xlabel('ระยะ X (ม.)', color='#8892a4', fontsize=8.5)
ax1.set_ylabel('ระยะ Y (ม.)', color='#8892a4', fontsize=8.5)
ax1.tick_params(colors='#8892a4', labelsize=8)
for spine in ax1.spines.values():
    spine.set_edgecolor('#2a3555')

# Colorbar
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax1, fraction=0.03, pad=0.02)
cbar.set_label('แรงในเสาเข็ม (ตัน)', color='#8892a4', fontsize=8)
cbar.ax.yaxis.set_tick_params(color='#8892a4', labelsize=7.5)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#8892a4')

# ── BAR CHART: Force per pile ──
ax2.set_facecolor('#0d1420')
pile_labels = [f'P{i+1}' for i in range(n)]
colors_bar = [cmap(norm(q)) for q in Qi]
bars = ax2.bar(pile_labels, Qi, color=colors_bar, edgecolor='#ffffff20', linewidth=0.8, zorder=3)

ax2.axhline(Qa_comp, color='#51cf66', linestyle='--', linewidth=1.5,
            label=f'Qa,comp = {Qa_comp:.0f} t', zorder=4)
ax2.axhline(-Qa_tens, color='#ff6b6b', linestyle='--', linewidth=1.5,
            label=f'Qa,tens = {Qa_tens:.0f} t', zorder=4)
ax2.axhline(0, color='#ffffff30', linewidth=0.5)

for bar, val in zip(bars, Qi):
    ypos = bar.get_height() + (max(Qi) - min(Qi)) * 0.03 if val >= 0 else val - (max(Qi) - min(Qi)) * 0.06
    ax2.text(bar.get_x() + bar.get_width()/2, ypos,
             f'{val:.1f}', ha='center', va='bottom' if val >= 0 else 'top',
             fontsize=7, color='#e8eaf0', fontweight='600', fontfamily='monospace')

ax2.set_title('แรงในแต่ละเสาเข็ม', color='#f5a623', fontsize=9.5, fontweight='700', pad=8)
ax2.set_ylabel('แรง (ตัน)', color='#8892a4', fontsize=8)
ax2.set_xlabel('เสาเข็มหมายเลข', color='#8892a4', fontsize=8)
ax2.tick_params(colors='#8892a4', labelsize=7.5)
ax2.legend(fontsize=7, facecolor='#1a2235', edgecolor='#2a3555',
           labelcolor='#e8eaf0', loc='upper right')
ax2.set_facecolor('#0d1420')
for spine in ax2.spines.values():
    spine.set_edgecolor('#2a3555')
ax2.grid(axis='y', color='#ffffff08', linewidth=0.5, zorder=0)

# ── UTILIZATION RATIO ──
ax3.set_facecolor('#0d1420')
util = []
util_colors = []
for q in Qi:
    if q >= 0:
        r = q / Qa_comp
    else:
        r = abs(q) / Qa_tens
    util.append(r)
    if r > 1.0:
        util_colors.append('#ff6b6b')
    elif r > 0.8:
        util_colors.append('#f5a623')
    else:
        util_colors.append('#51cf66')

bars2 = ax3.barh(pile_labels, util, color=util_colors,
                 edgecolor='#ffffff15', linewidth=0.8, zorder=3)
ax3.axvline(1.0, color='#ff6b6b', linestyle='--', linewidth=1.5,
            label='Limit = 1.0', zorder=4)
ax3.axvline(0.8, color='#f5a623', linestyle=':', linewidth=1.0,
            label='80% Limit', zorder=4)

for bar, val in zip(bars2, util):
    ax3.text(val + 0.01, bar.get_y() + bar.get_height()/2,
             f'{val*100:.1f}%', va='center', fontsize=7,
             color='#e8eaf0', fontweight='600', fontfamily='monospace')

ax3.set_title('อัตราส่วนใช้งาน (D/C Ratio)', color='#f5a623', fontsize=9.5,
              fontweight='700', pad=8)
ax3.set_xlabel('อัตราส่วน Q/Qa', color='#8892a4', fontsize=8)
ax3.tick_params(colors='#8892a4', labelsize=7.5)
ax3.legend(fontsize=7, facecolor='#1a2235', edgecolor='#2a3555',
           labelcolor='#e8eaf0')
ax3.set_xlim(0, max(1.2, max(util)*1.1))
for spine in ax3.spines.values():
    spine.set_edgecolor('#2a3555')
ax3.grid(axis='x', color='#ffffff08', linewidth=0.5, zorder=0)

fig.patch.set_facecolor('#0a0e1a')
plt.suptitle('ผลการวิเคราะห์กลุ่มเสาเข็ม | มยผ.1106-64',
             color='#8892a4', fontsize=9, y=0.02, fontfamily='Sarabun')
return fig
```

def plot_elevation(P, Mx, My, pile_depth, cap_thickness):
“”“แสดงภาพตัดด้านข้าง”””
fig, ax = plt.subplots(figsize=(8, 5), facecolor=’#0a0e1a’)
ax.set_facecolor(’#0d1420’)

```
total_depth = pile_depth + cap_thickness + 1.0
ax.set_xlim(-3, 3)
ax.set_ylim(-total_depth - 0.5, 4)
ax.set_aspect('equal')

# Ground line
ax.axhline(0, color='#8B7355', linewidth=2.5, zorder=3)
ax.fill_between([-3, 3], [-total_depth-0.5, -total_depth-0.5], [0, 0],
                color='#3d2e1e', alpha=0.4, zorder=0)
ax.fill_between([-3, 3], [0, 0], [4, 4],
                color='#0d1b2a', alpha=0.6, zorder=0)
ax.text(2.6, 0.1, 'ระดับดิน', color='#8B7355', fontsize=7.5, va='bottom')

# Pile cap
cap_top = 0.0
cap_bot = -cap_thickness
cap = FancyBboxPatch((-1.5, cap_bot), 3.0, cap_thickness,
                      boxstyle="square,pad=0",
                      facecolor='#2a3d5c', edgecolor='#4a9eff',
                      linewidth=1.5, zorder=4)
ax.add_patch(cap)
ax.text(0, cap_bot + cap_thickness/2, 'ฐานราก (Pile Cap)',
        ha='center', va='center', fontsize=8, color='#4a9eff',
        fontweight='600')

# Piles (2 representative)
for px in [-0.7, 0.7]:
    pile_rect = FancyBboxPatch((px-0.12, cap_bot - pile_depth), 0.24, pile_depth,
                                boxstyle="square,pad=0",
                                facecolor='#1e3a5f', edgecolor='#00d4aa',
                                linewidth=1.2, zorder=3)
    ax.add_patch(pile_rect)
    # Pile tip
    tip_y = cap_bot - pile_depth
    ax.annotate('', xy=(px, tip_y - 0.2), xytext=(px, tip_y),
                arrowprops=dict(arrowstyle='->', color='#00d4aa', lw=1.5))

ax.text(1.8, cap_bot - pile_depth/2, f'เสาเข็ม\nL = {pile_depth:.1f} ม.',
        ha='left', va='center', fontsize=7.5, color='#00d4aa', fontweight='600')

# Column
col = FancyBboxPatch((-0.25, cap_top), 0.5, 3.0,
                      boxstyle="round,pad=0.02",
                      facecolor='#2d3f5c', edgecolor='#f5a623',
                      linewidth=1.5, zorder=5)
ax.add_patch(col)
ax.text(0, cap_top + 1.5, 'เสา', ha='center', va='center',
        fontsize=8, color='#f5a623', fontweight='700')

# Load arrows
ax.annotate('', xy=(0, cap_top + 2.8), xytext=(0, 3.8),
            arrowprops=dict(arrowstyle='->', color='#4a9eff', lw=2.5))
ax.text(0.3, 3.3, f'P = {P:.0f} t', color='#4a9eff', fontsize=8,
        fontweight='700', va='center', fontfamily='monospace')

if abs(Mx) > 0.01:
    ax.annotate('', xy=(0.6, cap_top + 2.5), xytext=(0.1, cap_top + 3.0),
                arrowprops=dict(arrowstyle='->', color='#00d4aa', lw=2,
                                connectionstyle='arc3,rad=-0.4'))
    ax.text(1.0, cap_top + 2.7, f'Mx = {Mx:.0f} t·m',
            color='#00d4aa', fontsize=7.5, fontweight='600', fontfamily='monospace')

# Dimensions
ax.annotate('', xy=(-2.2, 0), xytext=(-2.2, cap_bot - pile_depth),
            arrowprops=dict(arrowstyle='<->', color='#8892a4', lw=1))
ax.text(-2.6, (cap_bot - pile_depth)/2, f'{pile_depth+cap_thickness:.1f} ม.',
        ha='center', va='center', fontsize=7.5, color='#8892a4',
        rotation=90)

ax.set_title('ภาพตัดด้านข้าง (Elevation View)', color='#f5a623',
             fontsize=10, fontweight='700', pad=10)
ax.set_xlabel('ระยะ (ม.)', color='#8892a4', fontsize=8)
ax.set_ylabel('ความลึก (ม.)', color='#8892a4', fontsize=8)
ax.tick_params(colors='#8892a4', labelsize=7.5)
for spine in ax.spines.values():
    spine.set_edgecolor('#2a3555')
ax.grid(color='#ffffff05', linewidth=0.5)

fig.patch.set_facecolor('#0a0e1a')
return fig
```

# ─────────────────────────────────────────────────────────

# SIDEBAR INPUTS

# ─────────────────────────────────────────────────────────

with st.sidebar:
st.markdown(”””
<div style='text-align:center; padding: 12px 0 20px 0;'>
<div style='font-size:2rem;'>🏗️</div>
<div style='font-size:0.95rem; font-weight:700; color:#f5a623;'>ออกแบบฐานราก</div>
<div style='font-size:0.72rem; color:#8892a4; margin-top:4px;'>มยผ.1106-64</div>
</div>
“””, unsafe_allow_html=True)

```
st.markdown("### 📐 แรงกระทำ (Load)")
P = st.number_input("น้ำหนักกด P (ตัน)", min_value=10.0, max_value=5000.0,
                    value=400.0, step=10.0, help="แรงกดแนวดิ่งรวม")
Mx = st.number_input("โมเมนต์ Mx (ตัน·เมตร)", min_value=-2000.0, max_value=2000.0,
                     value=150.0, step=5.0, help="โมเมนต์รอบแกน X")
My = st.number_input("โมเมนต์ My (ตัน·เมตร)", min_value=-2000.0, max_value=2000.0,
                     value=80.0, step=5.0, help="โมเมนต์รอบแกน Y")

st.markdown("---")
st.markdown("### 🔧 ข้อมูลเสาเข็ม")
pile_config = st.selectbox("รูปแบบการจัดเสาเข็ม", [
    "1 เสาเข็ม", "2 เสาเข็ม (แนวแกน X)", "2 เสาเข็ม (แนวแกน Y)",
    "3 เสาเข็ม (สามเหลี่ยม)", "4 เสาเข็ม (สี่เหลี่ยม)",
    "5 เสาเข็ม (X+Center)", "6 เสาเข็ม (2x3)",
    "8 เสาเข็ม (2x4)", "9 เสาเข็ม (3x3)",
], index=3)
spacing = st.slider("ระยะห่างเสาเข็ม (ม.)", 1.0, 4.0, 1.5, 0.1,
                    help="ระยะห่างระหว่างแนวเสาเข็ม (≥ 2.5D per มยผ.)")
pile_diam = st.number_input("ขนาดเสาเข็ม (ม.)", 0.20, 1.50, 0.35, 0.05)
pile_depth = st.number_input("ความยาวเสาเข็ม (ม.)", 5.0, 60.0, 21.0, 1.0)
cap_thickness = st.number_input("ความหนาฐานราก (ม.)", 0.5, 3.0, 1.0, 0.1)

st.markdown("---")
st.markdown("### ⚙️ ความสามารถรับแรงเสาเข็ม")
Qa_comp = st.number_input("Qa แบกรับ (ตัน/ต้น)", 10.0, 1000.0, 120.0, 5.0,
                          help="กำลังรับน้ำหนักประลัยด้านแบกรับ")
Qa_tens = st.number_input("Qa ดึง (ตัน/ต้น)", 0.0, 500.0, 40.0, 5.0,
                          help="กำลังรับแรงดึง (ถ้าไม่มีให้ใส่ 0)")

st.markdown("---")
st.markdown("### 📍 ความเยื้องศูนย์ (Eccentricity)")
st.caption("ตำแหน่งเสาเทียบกับจุดศูนย์ถ่วงกลุ่มเสาเข็ม")
ecc_x = st.number_input("ความเยื้อง ex (ม.)", -2.0, 2.0, 0.0, 0.05,
                        help="ระยะเยื้องศูนย์แนวแกน X")
ecc_y = st.number_input("ความเยื้อง ey (ม.)", -2.0, 2.0, 0.0, 0.05,
                        help="ระยะเยื้องศูนย์แนวแกน Y")

st.markdown("---")
calc_btn = st.button("🔍 คำนวณและออกแบบ", use_container_width=True)
```

# ─────────────────────────────────────────────────────────

# MAIN CONTENT

# ─────────────────────────────────────────────────────────

# Hero Banner

st.markdown(”””

<div class="hero-banner">
    <div class="hero-title">🏗️ ระบบออกแบบฐานรากรับโมเมนต์และตรวจสอบแรงในเสาเข็ม</div>
    <div class="hero-subtitle">กรณีเสาเข็มเยื้องศูนย์ — Eccentric Pile Group Analysis</div>
    <span class="hero-badge">มยผ.1106-64 | DPT Standard</span>
    <span class="hero-badge" style="margin-left:8px; background:rgba(0,212,170,0.1); border-color:rgba(0,212,170,0.3); color:#00d4aa;">
        Thai Geotechnical Engineering
    </span>
</div>
""", unsafe_allow_html=True)

# Tabs

tab1, tab2, tab3, tab4 = st.tabs([
“📊 ผลการคำนวณ”, “📐 ทฤษฎีและสูตร”, “📋 ตารางผล”, “ℹ️ วิธีใช้งาน”
])

# ─── TAB 1: RESULTS ───

with tab1:
if calc_btn or True:  # Auto calculate
pile_coords = generate_pile_layout(pile_config, spacing, ecc_x, ecc_y)
res = calculate_pile_forces(P, Mx, My, pile_coords)

```
    n = res["n"]
    Qi = res["Qi"]
    Qmax = res["Qmax"]
    Qmin = res["Qmin"]

    # Check minimum spacing
    min_spacing_req = 2.5 * pile_diam
    spacing_ok = spacing >= min_spacing_req

    # Status checks
    comp_ok = Qmax <= Qa_comp
    tens_ok = (Qmin >= 0) or (abs(Qmin) <= Qa_tens)
    all_ok = comp_ok and tens_ok

    # ── Summary Cards ──
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="result-item">
            <div class="result-label">จำนวนเสาเข็ม</div>
            <div class="result-value" style="color:#4a9eff;">{n}</div>
            <div class="result-unit">ต้น</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="result-item">
            <div class="result-label">แรงกดสูงสุด Qmax</div>
            <div class="result-value" style="color:{'#ff6b6b' if not comp_ok else '#51cf66'};">{Qmax:.1f}</div>
            <div class="result-unit">ตัน</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="result-item">
            <div class="result-label">แรงกดต่ำสุด Qmin</div>
            <div class="result-value" style="color:{'#ff6b6b' if Qmin < -Qa_tens else '#f5a623' if Qmin < 0 else '#51cf66'};">{Qmin:.1f}</div>
            <div class="result-unit">ตัน</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        ratio_max = Qmax / Qa_comp
        st.markdown(f"""
        <div class="result-item">
            <div class="result-label">D/C Ratio สูงสุด</div>
            <div class="result-value" style="color:{'#ff6b6b' if ratio_max>1 else '#f5a623' if ratio_max>0.8 else '#51cf66'};">{ratio_max:.2f}</div>
            <div class="result-unit">Q/Qa</div>
        </div>""", unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="result-item">
            <div class="result-label">สถานะรวม</div>
            <div class="result-value" style="font-size:1.2rem; color:{'#51cf66' if all_ok else '#ff6b6b'};">
                {'✅ ผ่าน' if all_ok else '❌ ไม่ผ่าน'}
            </div>
            <div class="result-unit">Overall</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Status Details ──
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.markdown(
            f'<div class="{"status-ok" if comp_ok else "status-fail"}">{"✓" if comp_ok else "✗"} แรงกด: {Qmax:.1f} / {Qa_comp:.0f} t = {Qmax/Qa_comp*100:.1f}%</div>',
            unsafe_allow_html=True)
    with col_s2:
        if Qmin < 0:
            st.markdown(
                f'<div class="{"status-ok" if tens_ok else "status-fail"}">{"✓" if tens_ok else "✗"} แรงดึง: {abs(Qmin):.1f} / {Qa_tens:.0f} t = {abs(Qmin)/Qa_tens*100:.1f}%</div>',
                unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-ok">✓ ไม่มีแรงดึง</div>', unsafe_allow_html=True)
    with col_s3:
        st.markdown(
            f'<div class="{"status-ok" if spacing_ok else "status-warn"}">{"✓" if spacing_ok else "⚠"} ระยะห่าง: {spacing:.2f} ม. (min {min_spacing_req:.2f} ม.)</div>',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main Visualization ──
    fig_plan = plot_pile_plan(pile_coords, Qi, Qa_comp, Qa_tens, ecc_x, ecc_y, P, Mx, My)
    st.pyplot(fig_plan, use_container_width=True)
    plt.close()

    # ── Elevation View ──
    col_el, col_info = st.columns([3, 2])
    with col_el:
        fig_elev = plot_elevation(P, Mx, My, pile_depth, cap_thickness)
        st.pyplot(fig_elev, use_container_width=True)
        plt.close()
    with col_info:
        st.markdown("""
        <div class="card">
            <div class="card-title"><span>📌</span>ข้อมูลสรุปการออกแบบ</div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="font-size:0.82rem; color:#8892a4; line-height:2.0;">
            <b style="color:#f5a623;">รูปแบบเสาเข็ม:</b> <span style="color:#e8eaf0; font-family:monospace;">{pile_config}</span><br>
            <b style="color:#f5a623;">ขนาดเสาเข็ม:</b> <span style="color:#e8eaf0; font-family:monospace;">∅{pile_diam*100:.0f} ซม.</span><br>
            <b style="color:#f5a623;">ความยาวเสาเข็ม:</b> <span style="color:#e8eaf0; font-family:monospace;">{pile_depth:.1f} ม.</span><br>
            <b style="color:#f5a623;">ระยะห่าง:</b> <span style="color:#e8eaf0; font-family:monospace;">{spacing:.2f} ม. = {spacing/pile_diam:.1f}D</span><br>
            <b style="color:#f5a623;">ความเยื้องศูนย์:</b> <span style="color:#e8eaf0; font-family:monospace;">ex={ecc_x:.2f}, ey={ecc_y:.2f} ม.</span><br>
            <b style="color:#f5a623;">Σxi²:</b> <span style="color:#e8eaf0; font-family:monospace;">{res['sum_xi2']:.3f} ม²</span><br>
            <b style="color:#f5a623;">Σyi²:</b> <span style="color:#e8eaf0; font-family:monospace;">{res['sum_yi2']:.3f} ม²</span><br>
            <b style="color:#f5a623;">P/n:</b> <span style="color:#e8eaf0; font-family:monospace;">{res['Q_axial']:.2f} ตัน/ต้น</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Eccentricity from moment
        if abs(P) > 0:
            ex_mom = My / P
            ey_mom = Mx / P
            st.markdown(f"""
            <div class="card" style="margin-top:8px;">
                <div class="card-title"><span>↔️</span>ความเยื้องศูนย์จากโมเมนต์</div>
                <div style="font-size:0.82rem; color:#8892a4; line-height:2.0;">
                    <b style="color:#00d4aa;">ex = My/P =</b> <span style="color:#e8eaf0; font-family:monospace;">{ex_mom:.3f} ม.</span><br>
                    <b style="color:#00d4aa;">ey = Mx/P =</b> <span style="color:#e8eaf0; font-family:monospace;">{ey_mom:.3f} ม.</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Warnings / Notes ──
    if not spacing_ok:
        st.warning(f"⚠️ ระยะห่างเสาเข็ม {spacing:.2f} ม. น้อยกว่าข้อกำหนด มยผ.1106-64 ขั้นต่ำ 2.5D = {min_spacing_req:.2f} ม.")
    if not comp_ok:
        st.error(f"❌ แรงกดสูงสุด {Qmax:.1f} ตัน เกินความสามารถรับแรง {Qa_comp:.0f} ตัน — ต้องเพิ่มจำนวนเสาเข็มหรือเพิ่มขนาด")
    if Qmin < 0 and not tens_ok:
        st.error(f"❌ แรงดึง {abs(Qmin):.1f} ตัน เกินความสามารถรับแรงดึง {Qa_tens:.0f} ตัน — ต้องพิจารณาการยึดหัวเสาเข็ม")
    if Qmin < 0 and tens_ok:
        st.warning(f"⚠️ เสาเข็มบางต้นรับแรงดึง ({Qmin:.1f} ตัน) ต้องตรวจสอบการเชื่อมต่อหัวเสาเข็มกับฐานราก")
```

# ─── TAB 2: THEORY ───

with tab2:
st.markdown(”””
<div class="card">
<div class="card-title"><span>📚</span>ทฤษฎีการคำนวณแรงในเสาเข็ม — มยผ.1106-64</div>
<div style="color:#8892a4; font-size:0.88rem; line-height:1.8;">
<p>มาตรฐาน <b style="color:#f5a623;">มยผ.1106-64</b> กำหนดให้คำนวณแรงในเสาเข็มแต่ละต้นจากสมการกระจายแรงเชิงเส้น
โดยสมมติว่าฐานรากมีความแข็งเกร็ง (Rigid Pile Cap) และเสาเข็มทุกต้นมีความแข็งเท่ากัน</p>
</div>
</div>
“””, unsafe_allow_html=True)

```
col_t1, col_t2 = st.columns(2)
with col_t1:
    st.markdown("""
    <div class="card">
        <div class="card-title"><span>🔢</span>สมการหลัก</div>
        <div class="formula-box">
            Qi = P/n ± (Mx·yi)/Σyi² ± (My·xi)/Σxi²
        </div>
        <div style="font-size:0.82rem; color:#8892a4; line-height:2.0; margin-top:12px;">
            <b style="color:#e8eaf0;">Qi</b> = แรงในเสาเข็มต้นที่ i (ตัน)<br>
            <b style="color:#e8eaf0;">P</b>  = แรงกดแนวดิ่งรวม (ตัน)<br>
            <b style="color:#e8eaf0;">n</b>  = จำนวนเสาเข็มทั้งหมด (ต้น)<br>
            <b style="color:#e8eaf0;">Mx</b> = โมเมนต์รอบแกน X (ตัน·เมตร)<br>
            <b style="color:#e8eaf0;">My</b> = โมเมนต์รอบแกน Y (ตัน·เมตร)<br>
            <b style="color:#e8eaf0;">xi</b> = ระยะห่างเสาเข็ม i จาก CG แนว X<br>
            <b style="color:#e8eaf0;">yi</b> = ระยะห่างเสาเข็ม i จาก CG แนว Y
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="card-title"><span>⚡</span>เงื่อนไขความเยื้องศูนย์</div>
        <div class="formula-box">
            ex = My / P  (ความเยื้องแกน X)<br>
            ey = Mx / P  (ความเยื้องแกน Y)<br><br>
            e_total = √(ex² + ey²)
        </div>
        <div style="font-size:0.82rem; color:#8892a4; margin-top:12px; line-height:1.8;">
            ค่าความเยื้องศูนย์รวมต้องอยู่ภายในแกนกลางของฐานราก (Kern)<br>
            สำหรับฐานรากสี่เหลี่ยม: <b style="color:#f5a623;">|ex| ≤ B/6, |ey| ≤ L/6</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_t2:
    st.markdown("""
    <div class="card">
        <div class="card-title"><span>✅</span>เกณฑ์การตรวจสอบ มยผ.1106-64</div>
        <div style="font-size:0.85rem; color:#8892a4; line-height:2.2;">
            <div style="border-left:3px solid #51cf66; padding-left:12px; margin-bottom:12px;">
                <b style="color:#51cf66;">1. แรงกดสูงสุด:</b><br>
                Qmax ≤ Qa (กำลังรับน้ำหนักประลัยด้านแบกรับ)
            </div>
            <div style="border-left:3px solid #ff6b6b; padding-left:12px; margin-bottom:12px;">
                <b style="color:#ff6b6b;">2. แรงดึงสูงสุด:</b><br>
                |Qmin| ≤ Qa,tension (ถ้ามีแรงดึง)
            </div>
            <div style="border-left:3px solid #f5a623; padding-left:12px; margin-bottom:12px;">
                <b style="color:#f5a623;">3. ระยะห่างเสาเข็ม:</b><br>
                s ≥ 2.5D (เสาเข็มตอก)<br>
                s ≥ 3.0D (เสาเข็มเจาะ)
            </div>
            <div style="border-left:3px solid #4a9eff; padding-left:12px;">
                <b style="color:#4a9eff;">4. ระยะขอบฐานราก:</b><br>
                ≥ 1.25D (จากขอบเสาเข็มถึงขอบฐานราก)
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <div class="card-title"><span>📏</span>ขั้นตอนการคำนวณ</div>
        <div style="font-size:0.82rem; color:#8892a4; line-height:2.0; counter-reset:step;">
            <div style="margin-bottom:8px; color:#e8eaf0;"><span style="color:#f5a623; font-weight:700;">① </span>กำหนดตำแหน่งเสาเข็มและหาจุดศูนย์ถ่วง (CG)</div>
            <div style="margin-bottom:8px; color:#e8eaf0;"><span style="color:#f5a623; font-weight:700;">② </span>คำนวณ xi, yi จาก CG</div>
            <div style="margin-bottom:8px; color:#e8eaf0;"><span style="color:#f5a623; font-weight:700;">③ </span>คำนวณ Σxi² และ Σyi²</div>
            <div style="margin-bottom:8px; color:#e8eaf0;"><span style="color:#f5a623; font-weight:700;">④ </span>คำนวณแรง Qi ในแต่ละต้น</div>
            <div style="margin-bottom:8px; color:#e8eaf0;"><span style="color:#f5a623; font-weight:700;">⑤ </span>ตรวจสอบ Qmax ≤ Qa และ Qmin ≥ -Qa,t</div>
            <div style="color:#e8eaf0;"><span style="color:#f5a623; font-weight:700;">⑥ </span>ตรวจสอบระยะห่างและข้อกำหนดอื่นๆ</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
```

# ─── TAB 3: TABLE ───

with tab3:
pile_coords = generate_pile_layout(pile_config, spacing, ecc_x, ecc_y)
res = calculate_pile_forces(P, Mx, My, pile_coords)
Qi = res[“Qi”]
n = res[“n”]

```
st.markdown("""
<div class="card">
    <div class="card-title"><span>📋</span>ตารางแรงในแต่ละเสาเข็ม</div>
""", unsafe_allow_html=True)

table_rows = ""
for i in range(n):
    x_coord, y_coord = pile_coords[i]
    xi_val = res["xi"][i]
    yi_val = res["yi"][i]
    qi_val = Qi[i]
    q_ax = res["Q_axial"]
    q_mx = res["Q_mx_contrib"][i]
    q_my = res["Q_my_contrib"][i]

    if qi_val >= 0:
        ratio = qi_val / Qa_comp
        cap_type = "กด"
    else:
        ratio = abs(qi_val) / Qa_tens if Qa_tens > 0 else 999
        cap_type = "ดึง"

    if ratio > 1.0:
        status_html = '<span class="status-fail">ไม่ผ่าน</span>'
        qi_cls = "td-critical"
    elif ratio > 0.8:
        status_html = '<span class="status-warn">ระวัง</span>'
        qi_cls = "td-warn"
    else:
        status_html = '<span class="status-ok">ผ่าน</span>'
        qi_cls = "td-ok"

    table_rows += f"""
    <tr>
        <td><b style="color:#4a9eff;">P{i+1}</b></td>
        <td>{x_coord:.3f}</td>
        <td>{y_coord:.3f}</td>
        <td>{xi_val:.3f}</td>
        <td>{yi_val:.3f}</td>
        <td>{q_ax:.2f}</td>
        <td>{q_mx:+.2f}</td>
        <td>{q_my:+.2f}</td>
        <td class="{qi_cls}"><b>{qi_val:.2f}</b></td>
        <td>{cap_type}</td>
        <td>{ratio*100:.1f}%</td>
        <td>{status_html}</td>
    </tr>"""

st.markdown(f"""
<table class="data-table">
    <thead>
        <tr>
            <th>เสาเข็ม</th>
            <th>X (ม.)</th>
            <th>Y (ม.)</th>
            <th>xi (ม.)</th>
            <th>yi (ม.)</th>
            <th>P/n (t)</th>
            <th>±Mx·y/Σy² (t)</th>
            <th>±My·x/Σx² (t)</th>
            <th>Qi (ตัน)</th>
            <th>ประเภท</th>
            <th>D/C (%)</th>
            <th>สถานะ</th>
        </tr>
    </thead>
    <tbody>
        {table_rows}
    </tbody>
</table>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Summary table
st.markdown("""
<div class="card">
    <div class="card-title"><span>📊</span>ตารางสรุปค่าพารามิเตอร์</div>
""", unsafe_allow_html=True)

xs_all = [c[0] for c in pile_coords]
ys_all = [c[1] for c in pile_coords]
xi_arr = res["xi"]
yi_arr = res["yi"]

param_rows = "".join([
    f"<tr><td><b style='color:#4a9eff;'>P{i+1}</b></td>"
    f"<td>{xi_arr[i]:.3f}</td>"
    f"<td>{yi_arr[i]:.3f}</td>"
    f"<td>{xi_arr[i]**2:.4f}</td>"
    f"<td>{yi_arr[i]**2:.4f}</td></tr>"
    for i in range(n)
])
param_rows += f"""
<tr style="background:rgba(245,166,35,0.1);">
    <td><b style="color:#f5a623;">รวม</b></td>
    <td><b style="color:#f5a623;">{sum(xi_arr):.4f}</b></td>
    <td><b style="color:#f5a623;">{sum(yi_arr):.4f}</b></td>
    <td><b style="color:#f5a623;">{res['sum_xi2']:.4f}</b></td>
    <td><b style="color:#f5a623;">{res['sum_yi2']:.4f}</b></td>
</tr>"""

st.markdown(f"""
<table class="data-table">
    <thead>
        <tr><th>เสาเข็ม</th><th>xi (ม.)</th><th>yi (ม.)</th><th>xi² (ม²)</th><th>yi² (ม²)</th></tr>
    </thead>
    <tbody>{param_rows}</tbody>
</table>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
```

# ─── TAB 4: HELP ───

with tab4:
col_h1, col_h2 = st.columns(2)
with col_h1:
st.markdown(”””
<div class="card">
<div class="card-title"><span>📖</span>วิธีใช้งานโปรแกรม</div>
<div style="font-size:0.85rem; color:#8892a4; line-height:2.0;">
<div style="color:#e8eaf0; margin-bottom:8px;"><b style="color:#f5a623;">1. กรอกแรงกระทำ</b><br>
ใส่ค่า P, Mx, My ที่ฐานเสา ค่าโมเมนต์ใช้เครื่องหมาย ± ได้</div>
<div style="color:#e8eaf0; margin-bottom:8px;"><b style="color:#f5a623;">2. เลือกรูปแบบเสาเข็ม</b><br>
เลือกจากรูปแบบที่กำหนด และตั้งค่าระยะห่าง</div>
<div style="color:#e8eaf0; margin-bottom:8px;"><b style="color:#f5a623;">3. กรอกข้อมูลเสาเข็ม</b><br>
ขนาด ความยาว และความสามารถรับแรง (Qa)</div>
<div style="color:#e8eaf0; margin-bottom:8px;"><b style="color:#f5a623;">4. ตั้งค่าความเยื้องศูนย์</b><br>
กรณีเสาไม่ตรงกับจุดศูนย์ถ่วงกลุ่มเสาเข็ม</div>
<div style="color:#e8eaf0;"><b style="color:#f5a623;">5. ดูผลการคำนวณ</b><br>
ระบบแสดงผลอัตโนมัติพร้อมแผนภาพ</div>
</div>
</div>
“””, unsafe_allow_html=True)

```
with col_h2:
    st.markdown("""
    <div class="card">
        <div class="card-title"><span>⚠️</span>ข้อควรระวัง</div>
        <div style="font-size:0.85rem; color:#8892a4; line-height:2.0;">
            <div style="border-left:2px solid #ff6b6b; padding-left:10px; margin-bottom:10px; color:#e8eaf0;">
                โปรแกรมนี้คำนวณตามทฤษฎีฐานรากแข็งเกร็ง (Rigid Cap) ไม่คำนึงถึงการทรุดตัวเชิงอนุพันธ์
            </div>
            <div style="border-left:2px solid #f5a623; padding-left:10px; margin-bottom:10px; color:#e8eaf0;">
                ค่า Qa ที่ใส่ต้องเป็นค่าที่ผ่านการทดสอบ Static Load Test หรือ Dynamic Load Test แล้ว
            </div>
            <div style="border-left:2px solid #4a9eff; padding-left:10px; margin-bottom:10px; color:#e8eaf0;">
                ต้องตรวจสอบผลเชิง Lateral Force แยกต่างหากตาม มยผ.1106-64 ข้อ 5
            </div>
            <div style="border-left:2px solid #51cf66; padding-left:10px; color:#e8eaf0;">
                วิศวกรผู้รับผิดชอบต้องพิจารณาปัจจัยอื่นๆ เช่น Negative Skin Friction, Group Effect ฯลฯ
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="card">
    <div class="card-title"><span>📜</span>อ้างอิงมาตรฐาน</div>
    <div style="font-size:0.85rem; color:#8892a4; line-height:2.0; display:grid; grid-template-columns:1fr 1fr; gap:8px;">
        <div style="color:#e8eaf0;"><b style="color:#f5a623;">มยผ.1106-64</b> — มาตรฐานการออกแบบและก่อสร้างฐานรากเสาเข็ม</div>
        <div style="color:#e8eaf0;"><b style="color:#f5a623;">มยผ.1101-64</b> — มาตรฐานการออกแบบอาคารคอนกรีตเสริมเหล็ก</div>
        <div style="color:#e8eaf0;"><b style="color:#f5a623;">ACI 318-19</b> — Building Code Requirements for Structural Concrete</div>
        <div style="color:#e8eaf0;"><b style="color:#f5a623;">ASCE 7-22</b> — Minimum Design Loads and Associated Criteria</div>
    </div>
</div>
""", unsafe_allow_html=True)
```

# Footer

st.markdown(”””

<div style="text-align:center; padding:24px 0 8px; color:#3d4a5c; font-size:0.75rem; border-top:1px solid #1a2235; margin-top:32px;">
    🏗️ ระบบออกแบบฐานรากเสาเข็ม | มยผ.1106-64 | พัฒนาด้วย Python + Streamlit<br>
    <span style="color:#2a3555;">สงวนลิขสิทธิ์ — ใช้เพื่อการศึกษาและวิศวกรรมเท่านั้น</span>
</div>
""", unsafe_allow_html=True)