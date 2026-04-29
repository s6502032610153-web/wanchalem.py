"""
โปรแกรมคำนวณและออกแบบฐานรากเสาเข็มแบบมีโมเมนต์จากการเยื้องศูนย์
(Eccentric Pile Foundation Design)
ตามมาตรฐาน มยผ.1106-64
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Eccentric Pile Foundation | มยผ.1106-64",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — industrial-blueprint aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@400;600;700&family=Barlow:wght@400;500&display=swap');

:root {
    --bg:       #0d1117;
    --surface:  #161b22;
    --border:   #30363d;
    --accent:   #f0a500;
    --accent2:  #58a6ff;
    --ok:       #3fb950;
    --warn:     #e3b341;
    --fail:     #f85149;
    --text:     #e6edf3;
    --muted:    #8b949e;
    --mono:     'Share Tech Mono', monospace;
    --head:     'Barlow Condensed', sans-serif;
    --body:     'Barlow', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--body);
    background-color: var(--bg);
    color: var(--text);
}

/* ── Title ── */
.app-title {
    font-family: var(--head);
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: var(--accent);
    text-transform: uppercase;
    border-bottom: 2px solid var(--accent);
    padding-bottom: 0.3rem;
    margin-bottom: 0.2rem;
}
.app-sub {
    font-family: var(--mono);
    font-size: 0.82rem;
    color: var(--muted);
    letter-spacing: 0.12em;
    margin-bottom: 1.5rem;
}

/* ── Section headers ── */
.sec-header {
    font-family: var(--head);
    font-size: 1.15rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: var(--accent2);
    text-transform: uppercase;
    border-left: 3px solid var(--accent2);
    padding-left: 0.6rem;
    margin: 1.2rem 0 0.6rem;
}

/* ── Result cards ── */
.result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.result-label {
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.result-value {
    font-family: var(--mono);
    font-size: 1.6rem;
    font-weight: bold;
    color: var(--accent);
}

/* ── Status badge ── */
.badge-ok   { background:#0d2b12; color:var(--ok);   border:1px solid var(--ok);   border-radius:4px; padding:2px 10px; font-family:var(--mono); font-size:0.85rem; font-weight:bold; }
.badge-warn { background:#2b2006; color:var(--warn); border:1px solid var(--warn); border-radius:4px; padding:2px 10px; font-family:var(--mono); font-size:0.85rem; font-weight:bold; }
.badge-fail { background:#2b0a0a; color:var(--fail); border:1px solid var(--fail); border-radius:4px; padding:2px 10px; font-family:var(--mono); font-size:0.85rem; font-weight:bold; }

/* ── Equation block ── */
.eq-box {
    background: #0d1117;
    border: 1px solid #30363d;
    border-left: 3px solid var(--accent);
    border-radius: 4px;
    padding: 0.5rem 1rem;
    font-family: var(--mono);
    font-size: 0.85rem;
    color: var(--accent);
    margin: 0.4rem 0 0.8rem;
    letter-spacing: 0.05em;
}

/* ── Streamlit widget overrides ── */
.stNumberInput > label, .stTextInput > label, .stSelectbox > label {
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
    color: var(--muted) !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
div[data-testid="stNumberInput"] input {
    background: #0d1117 !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
}
.stButton > button {
    background: var(--accent) !important;
    color: #000 !important;
    font-family: var(--head) !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.55rem 2rem !important;
}
.stButton > button:hover {
    background: #ffbc30 !important;
}
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 6px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: var(--head) !important;
    color: var(--accent) !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PileGroup:
    """เก็บข้อมูลกลุ่มเสาเข็ม"""
    x: List[float]
    y: List[float]
    n: int

    @property
    def positions(self) -> List[Tuple[float, float]]:
        return list(zip(self.x, self.y))


@dataclass
class FoundationInput:
    """ข้อมูล input ทั้งหมดของฐานราก"""
    P: float           # แรงกด (kN)
    Mx: float          # โมเมนต์รอบแกน x (kN·m)
    My: float          # โมเมนต์รอบแกน y (kN·m)
    piles: PileGroup
    Q_allow: float     # กำลังรับน้ำหนักเสาเข็ม (kN/ต้น)
    B: float           # ความกว้างฐานราก (m)
    L: float           # ความยาวฐานราก (m)
    t: float           # ความหนาฐานราก (m)
    fc: float          # กำลังอัดคอนกรีต (MPa)
    fy: float          # กำลังคราก เหล็ก (MPa)


# ─────────────────────────────────────────────────────────────────────────────
# Engineering functions (มยผ.1106-64)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_centroid(piles: PileGroup) -> Tuple[float, float]:
    """
    คำนวณจุดศูนย์กลาง (centroid) ของกลุ่มเสาเข็ม
    x̄ = ΣxᵢΣ / n , ȳ = Σyᵢ / n
    """
    cx = np.mean(piles.x)
    cy = np.mean(piles.y)
    return cx, cy


def calculate_moment_of_inertia(piles: PileGroup, cx: float, cy: float) -> Tuple[float, float]:
    """
    คำนวณโมเมนต์ความเฉื่อยของกลุ่มเสาเข็ม
    Ix = Σ(yᵢ - ȳ)²   (m²·ต้น)
    Iy = Σ(xᵢ - x̄)²
    """
    xi = np.array(piles.x)
    yi = np.array(piles.y)
    Ix = np.sum((yi - cy) ** 2)
    Iy = np.sum((xi - cx) ** 2)
    return Ix, Iy


def calculate_pile_forces(
    P: float, Mx: float, My: float,
    piles: PileGroup,
    cx: float, cy: float,
    Ix: float, Iy: float
) -> np.ndarray:
    """
    คำนวณแรงที่กระทำต่อเสาเข็มแต่ละต้น (มยผ.1106-64 ข้อ 4)
    Pᵢ = P/n  ±  Mx·yᵢ'/Ix  ±  My·xᵢ'/Iy
    โดย xᵢ' = xᵢ − x̄  ,  yᵢ' = yᵢ − ȳ
    """
    n = piles.n
    xi = np.array(piles.x) - cx      # ระยะห่างจาก centroid แกน x
    yi = np.array(piles.y) - cy      # ระยะห่างจาก centroid แกน y

    P_axial = P / n                   # แรงอัดเฉลี่ย

    # แรงเนื่องจากโมเมนต์ (หาร Ix, Iy; ป้องกัน division-by-zero)
    term_Mx = (Mx * yi / Ix) if Ix > 1e-12 else np.zeros(n)
    term_My = (My * xi / Iy) if Iy > 1e-12 else np.zeros(n)

    forces = P_axial + term_Mx + term_My
    return forces


def check_capacity(forces: np.ndarray, Q_allow: float) -> Tuple[bool, List[int], List[int]]:
    """
    ตรวจสอบความสามารถรับแรงของเสาเข็ม
    - แรงอัด ≤ Q_allow  → OK
    - แรงดึง (< 0)       → WARNING
    Returns: (all_ok, failed_indices, tension_indices)
    """
    failed  = [i for i, f in enumerate(forces) if f > Q_allow]
    tension = [i for i, f in enumerate(forces) if f < 0]
    all_ok  = (len(failed) == 0) and (len(tension) == 0)
    return all_ok, failed, tension


# ─────────────────────────────────────────────────────────────────────────────
# Plot function
# ─────────────────────────────────────────────────────────────────────────────

def plot_pile_group(inp: FoundationInput, forces: np.ndarray, cx: float, cy: float):
    """
    วาดแผนผังกลุ่มเสาเข็มพร้อมแสดงแรงแต่ละต้น
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    # ── ฐานรากรูปสี่เหลี่ยม ──
    rect_x = cx - inp.B / 2
    rect_y = cy - inp.L / 2
    rect = patches.Rectangle(
        (rect_x, rect_y), inp.B, inp.L,
        linewidth=1.5, edgecolor='#30363d',
        facecolor='#161b22', zorder=1, linestyle='--'
    )
    ax.add_patch(rect)

    # ── Color map ตามขนาดแรง ──
    f_min, f_max = forces.min(), forces.max()
    norm = plt.Normalize(vmin=f_min, vmax=f_max)
    cmap = LinearSegmentedColormap.from_list(
        'pile_cmap', ['#58a6ff', '#3fb950', '#f0a500', '#f85149']
    )

    for i, (xi, yi) in enumerate(zip(inp.piles.x, inp.piles.y)):
        color = cmap(norm(forces[i]))
        # วงกลมเสาเข็ม
        circle = plt.Circle((xi, yi), 0.08, color=color, zorder=3, linewidth=2)
        ax.add_patch(circle)
        # เลขกำกับ
        ax.annotate(
            f"P{i+1}\n{forces[i]:.1f} kN",
            (xi, yi), textcoords="offset points", xytext=(8, 8),
            fontsize=8, color='#e6edf3',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#161b22',
                      edgecolor=color, linewidth=1, alpha=0.85),
            zorder=4
        )

    # ── centroid ──
    ax.plot(cx, cy, 'x', color='#f0a500', markersize=12, markeredgewidth=2.5, zorder=5)
    ax.annotate(
        f" Centroid\n ({cx:.2f}, {cy:.2f})",
        (cx, cy), fontsize=8, color='#f0a500', fontfamily='monospace'
    )

    # ── colorbar ──
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Pile Force (kN)', color='#8b949e', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='#8b949e')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#8b949e', fontsize=8)

    # ── axis style ──
    ax.set_xlabel('X (m)', color='#8b949e', fontfamily='monospace')
    ax.set_ylabel('Y (m)', color='#8b949e', fontfamily='monospace')
    ax.set_title('แผนผังกลุ่มเสาเข็ม  —  Pile Force Distribution',
                 color='#e6edf3', fontsize=11, fontweight='bold', pad=10)
    ax.tick_params(colors='#8b949e', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.set_aspect('equal')
    ax.grid(True, color='#21262d', linewidth=0.5, linestyle=':')

    # ── padding ──
    margin = max(inp.B, inp.L) * 0.25 + 0.5
    ax.set_xlim(cx - inp.B/2 - margin, cx + inp.B/2 + margin)
    ax.set_ylim(cy - inp.L/2 - margin, cy + inp.L/2 + margin)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# UI — Sidebar: all inputs
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🏗️ มยผ.1106-64")
    st.markdown("**Eccentric Pile Foundation**")
    st.markdown("---")

    # ── แรงกระทำ ──
    st.markdown("**แรงกระทำ / Applied Loads**")
    P  = st.number_input("P — แรงอัด (kN)",   value=3000.0, step=100.0, format="%.1f")
    Mx = st.number_input("Mx — โมเมนต์แกน X (kN·m)", value=200.0, step=10.0, format="%.1f")
    My = st.number_input("My — โมเมนต์แกน Y (kN·m)", value=150.0, step=10.0, format="%.1f")
    st.markdown("---")

    # ── เสาเข็ม ──
    st.markdown("**เสาเข็ม / Piles**")
    n_piles   = st.number_input("จำนวนเสาเข็ม (ต้น)", min_value=1, max_value=20, value=6, step=1)
    Q_allow   = st.number_input("กำลังรับน้ำหนักต่อต้น Q_allow (kN)", value=600.0, step=50.0, format="%.1f")

    # ── ตำแหน่งเสาเข็ม ──
    st.markdown("**ตำแหน่งเสาเข็ม (xᵢ, yᵢ) หน่วย: เมตร**")

    # Default positions สำหรับ 6 ต้น
    default_positions = [
        (-0.9, -0.9), (0.0, -0.9), (0.9, -0.9),
        (-0.9,  0.9), (0.0,  0.9), (0.9,  0.9),
    ]
    xs, ys = [], []
    for i in range(n_piles):
        dx = default_positions[i][0] if i < len(default_positions) else 0.0
        dy = default_positions[i][1] if i < len(default_positions) else 0.0
        c1, c2 = st.columns(2)
        with c1:
            xi = st.number_input(f"x{i+1}", value=dx, step=0.1, key=f"x{i}", format="%.2f")
        with c2:
            yi = st.number_input(f"y{i+1}", value=dy, step=0.1, key=f"y{i}", format="%.2f")
        xs.append(xi); ys.append(yi)

    st.markdown("---")

    # ── ฐานราก ──
    st.markdown("**ฐานราก / Foundation**")
    B = st.number_input("ความกว้าง B (m)", value=3.0, step=0.1, format="%.1f")
    L = st.number_input("ความยาว L (m)",  value=3.0, step=0.1, format="%.1f")
    t = st.number_input("ความหนา t (m)",  value=0.8, step=0.05, format="%.2f")
    st.markdown("---")

    # ── วัสดุ ──
    st.markdown("**คุณสมบัติวัสดุ / Materials**")
    fc = st.number_input("fc' (MPa)", value=28.0, step=1.0, format="%.1f")
    fy = st.number_input("fy (MPa)",  value=392.0, step=10.0, format="%.1f")
    st.markdown("---")

    calc_btn = st.button("▶  คำนวณ / Calculate", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# UI — Main area
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="app-title">⚙ Eccentric Pile Foundation Designer</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">มาตรฐาน มยผ.1106-64  ·  Pile Group with Eccentric Loading  ·  v1.0</div>', unsafe_allow_html=True)

# ── สูตรอ้างอิง ──
with st.expander("📐 สูตรคำนวณอ้างอิง (มยผ.1106-64)"):
    st.markdown('<div class="eq-box">Pᵢ = P/n  ±  (Mx · yᵢ\') / Ix  ±  (My · xᵢ\') / Iy</div>', unsafe_allow_html=True)
    st.markdown('<div class="eq-box">Ix = Σ(yᵢ − ȳ)²   |   Iy = Σ(xᵢ − x̄)²</div>', unsafe_allow_html=True)
    st.markdown('<div class="eq-box">xᵢ\' = xᵢ − x̄   |   yᵢ\' = yᵢ − ȳ   (ระยะจาก centroid)</div>', unsafe_allow_html=True)

if not calc_btn:
    st.info("← กรอกข้อมูลในแถบด้านซ้ายและกด **คำนวณ** เพื่อแสดงผล")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Calculation
# ─────────────────────────────────────────────────────────────────────────────

piles = PileGroup(x=xs, y=ys, n=n_piles)
inp   = FoundationInput(P=P, Mx=Mx, My=My, piles=piles,
                         Q_allow=Q_allow, B=B, L=L, t=t, fc=fc, fy=fy)

# 1) Centroid
cx, cy = calculate_centroid(piles)

# 2) Moment of inertia
Ix, Iy = calculate_moment_of_inertia(piles, cx, cy)

# 3) Pile forces
forces = calculate_pile_forces(P, Mx, My, piles, cx, cy, Ix, Iy)

# 4) Check capacity
all_ok, failed_idx, tension_idx = check_capacity(forces, Q_allow)

# Derived values
P_max   = forces.max()
P_min   = forces.min()
i_max   = int(np.argmax(forces))
i_min   = int(np.argmin(forces))
P_avg   = P / n_piles
e_x     = My / P if P != 0 else 0   # ความเยื้องศูนย์
e_y     = Mx / P if P != 0 else 0

# ─────────────────────────────────────────────────────────────────────────────
# Output layout
# ─────────────────────────────────────────────────────────────────────────────

col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    # ── Summary metrics ──
    st.markdown('<div class="sec-header">ผลสรุป / Summary</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Centroid X (m)", f"{cx:.3f}")
    m2.metric("Centroid Y (m)", f"{cy:.3f}")
    m3.metric("P เฉลี่ย (kN)", f"{P_avg:.1f}")

    m4, m5, m6 = st.columns(3)
    m4.metric("Ix (m²·ต้น)", f"{Ix:.4f}")
    m5.metric("Iy (m²·ต้น)", f"{Iy:.4f}")
    m6.metric("eₓ / e_y (m)", f"{e_x:.3f} / {e_y:.3f}")

    # ── Status ──
    st.markdown('<div class="sec-header">สถานะการออกแบบ / Design Status</div>', unsafe_allow_html=True)

    if all_ok:
        st.markdown('<span class="badge-ok">✔  PASS — ผ่านเกณฑ์ทุกข้อ</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-fail">✘  FAIL — ไม่ผ่านเกณฑ์</span>', unsafe_allow_html=True)

    if tension_idx:
        pile_ids = ", ".join([f"P{i+1}" for i in tension_idx])
        st.markdown(f'<br><span class="badge-warn">⚠  WARNING — เสาเข็ม {pile_ids} รับแรงดึง (Tension)</span>',
                    unsafe_allow_html=True)

    if failed_idx:
        pile_ids = ", ".join([f"P{i+1}" for i in failed_idx])
        st.markdown(f'<br><span class="badge-fail">✘  เสาเข็ม {pile_ids} รับแรงเกิน Q_allow = {Q_allow:.1f} kN</span>',
                    unsafe_allow_html=True)

    # ── Max / Min ──
    st.markdown('<div class="sec-header">แรงสูงสุด – ต่ำสุด</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        status_max = "✔ OK" if P_max <= Q_allow else "✘ OVER"
        color_max  = "#3fb950" if P_max <= Q_allow else "#f85149"
        st.markdown(f"""
        <div class="result-card">
          <div class="result-label">แรงสูงสุด — เสาเข็ม P{i_max+1}</div>
          <div class="result-value" style="color:{color_max}">{P_max:.2f} kN</div>
          <div style="font-family:var(--mono);font-size:0.75rem;color:{color_max}">{status_max}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        t_label = "⚠ TENSION" if P_min < 0 else "✔ OK"
        t_color = "#e3b341" if P_min < 0 else "#3fb950"
        st.markdown(f"""
        <div class="result-card">
          <div class="result-label">แรงต่ำสุด — เสาเข็ม P{i_min+1}</div>
          <div class="result-value" style="color:{t_color}">{P_min:.2f} kN</div>
          <div style="font-family:var(--mono);font-size:0.75rem;color:{t_color}">{t_label}</div>
        </div>""", unsafe_allow_html=True)

    # ── Pile force table ──
    st.markdown('<div class="sec-header">แรงในเสาเข็มแต่ละต้น</div>', unsafe_allow_html=True)

    xi_arr = np.array(piles.x)
    yi_arr = np.array(piles.y)

    rows = []
    for i in range(n_piles):
        xi_c = xi_arr[i] - cx
        yi_c = yi_arr[i] - cy
        term_mx = (Mx * yi_c / Ix) if Ix > 1e-12 else 0.0
        term_my = (My * xi_c / Iy) if Iy > 1e-12 else 0.0
        fi = forces[i]

        if fi < 0:
            status = "⚠ TENSION"
        elif fi > Q_allow:
            status = "✘ OVER"
        else:
            status = "✔ OK"

        rows.append({
            "เสาเข็ม": f"P{i+1}",
            "xᵢ (m)": f"{xi_arr[i]:.3f}",
            "yᵢ (m)": f"{yi_arr[i]:.3f}",
            "xᵢ' (m)": f"{xi_c:.3f}",
            "yᵢ' (m)": f"{yi_c:.3f}",
            "P/n (kN)": f"{P_avg:.2f}",
            "±Mx·y/Ix": f"{term_mx:+.2f}",
            "±My·x/Iy": f"{term_my:+.2f}",
            "Pᵢ (kN)": f"{fi:.2f}",
            "สถานะ": status,
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


with col_right:
    st.markdown('<div class="sec-header">แผนผังกลุ่มเสาเข็ม</div>', unsafe_allow_html=True)
    fig = plot_pile_group(inp, forces, cx, cy)
    st.pyplot(fig, use_container_width=True)

    # ── ข้อมูลวัสดุ ──
    st.markdown('<div class="sec-header">คุณสมบัติวัสดุ / Materials</div>', unsafe_allow_html=True)
    mat_c1, mat_c2 = st.columns(2)
    mat_c1.metric("fc' (MPa)", f"{fc:.1f}")
    mat_c1.metric("fy (MPa)",  f"{fy:.1f}")
    mat_c2.metric("B × L × t (m)", f"{B:.1f} × {L:.1f} × {t:.2f}")
    Wf = B * L * t * 24   # น้ำหนักฐานราก (kN) สมมติ γ = 24 kN/m³
    mat_c2.metric("น้ำหนักฐานราก (kN)", f"{Wf:.1f}")

    # ── Stability note ──
    st.markdown('<div class="sec-header">หมายเหตุ / Notes</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="result-card" style="font-family:var(--mono);font-size:0.78rem;color:#8b949e;line-height:1.7">
    • จำนวนเสาเข็ม : <span style="color:#e6edf3">{n_piles} ต้น</span><br>
    • Q_allow ต่อต้น : <span style="color:#e6edf3">{Q_allow:.1f} kN</span><br>
    • กำลังรวมกลุ่ม : <span style="color:#e6edf3">{n_piles * Q_allow:.1f} kN</span><br>
    • แรงกด P รวม : <span style="color:#e6edf3">{P:.1f} kN</span><br>
    • Utilization สูงสุด : <span style="color:{'#f85149' if P_max/Q_allow > 1 else '#3fb950'}">{P_max/Q_allow*100:.1f} %</span><br>
    • ความเยื้องศูนย์ eₓ : <span style="color:#e6edf3">{e_x:.3f} m</span> ,  e_y : <span style="color:#e6edf3">{e_y:.3f} m</span>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<span style="font-family:\'Share Tech Mono\',monospace;font-size:0.72rem;color:#484f58">'
    'มยผ.1106-64  ·  Eccentric Pile Foundation  ·  Built with Python + Streamlit'
    '</span>',
    unsafe_allow_html=True
)
