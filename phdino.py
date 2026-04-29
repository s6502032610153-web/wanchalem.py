"""
=============================================================================
โปรแกรมคำนวณและออกแบบฐานรากเสาเข็มแบบมีโมเมนต์จากการเยื้องศูนย์
Eccentric Pile Foundation Design Calculator
มาตรฐาน: มยผ.1106-64 (กรมโยธาธิการและผังเมือง)
=============================================================================
"""

import math
import sys

# ตรวจสอบว่ามี matplotlib หรือไม่
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[INFO] ไม่พบ matplotlib — จะไม่แสดงกราฟ (ติดตั้งด้วย: pip install matplotlib)")


# =============================================================================
# ฟังก์ชันที่ 1: คำนวณ centroid ของกลุ่มเสาเข็ม
# =============================================================================
def calculate_centroid(pile_coords: list[tuple]) -> tuple:
    """
    คำนวณจุดศูนย์กลาง (centroid) ของกลุ่มเสาเข็ม
    สมมติว่าเสาเข็มทุกต้นมีน้ำหนักเท่ากัน

    Args:
        pile_coords: รายการ [(x1,y1), (x2,y2), ...] ตำแหน่งเสาเข็ม (เมตร)

    Returns:
        (xc, yc): พิกัด centroid (เมตร)
    """
    n = len(pile_coords)
    xc = sum(x for x, y in pile_coords) / n
    yc = sum(y for x, y in pile_coords) / n
    return xc, yc


# =============================================================================
# ฟังก์ชันที่ 2: คำนวณ moment of inertia ของกลุ่มเสาเข็ม
# =============================================================================
def calculate_moment_of_inertia(pile_coords: list[tuple], centroid: tuple) -> tuple:
    """
    คำนวณ moment of inertia (Ix, Iy) ของกลุ่มเสาเข็มรอบ centroid
    ใช้สูตร: Ix = Σ(yi²)  , Iy = Σ(xi²)
    โดย xi, yi คือระยะห่างจาก centroid

    Args:
        pile_coords: รายการตำแหน่งเสาเข็ม [(x1,y1), ...]
        centroid: จุดศูนย์กลาง (xc, yc)

    Returns:
        (Ix, Iy): หน่วย เมตร²
    """
    xc, yc = centroid
    Ix = sum((y - yc) ** 2 for x, y in pile_coords)  # รอบแกน x (ใช้ระยะ y)
    Iy = sum((x - xc) ** 2 for x, y in pile_coords)  # รอบแกน y (ใช้ระยะ x)
    return Ix, Iy


# =============================================================================
# ฟังก์ชันที่ 3: คำนวณแรงที่ตกในเสาเข็มแต่ละต้น
# =============================================================================
def calculate_pile_forces(
    P: float,
    Mx: float,
    My: float,
    pile_coords: list[tuple],
    centroid: tuple,
    Ix: float,
    Iy: float,
) -> list[float]:
    """
    คำนวณแรงในเสาเข็มแต่ละต้น ตามสูตร มยผ.1106-64:
        Pi = P/n ± (Mx * yi / Ix) ± (My * xi / Iy)

    โดย:
        P   = แรงอัดแนวแกน (kN)
        Mx  = โมเมนต์รอบแกน x (kN·m)
        My  = โมเมนต์รอบแกน y (kN·m)
        xi  = ระยะห่างของเสาเข็มต้นที่ i จาก centroid ในแกน x
        yi  = ระยะห่างของเสาเข็มต้นที่ i จาก centroid ในแกน y
        Ix  = Σyi² (m²)
        Iy  = Σxi² (m²)

    Args:
        P, Mx, My: แรงและโมเมนต์
        pile_coords: ตำแหน่งเสาเข็ม
        centroid: (xc, yc)
        Ix, Iy: moment of inertia

    Returns:
        รายการแรงในแต่ละเสาเข็ม [P1, P2, ...] (kN)
    """
    xc, yc = centroid
    n = len(pile_coords)
    forces = []

    for x, y in pile_coords:
        xi = x - xc  # ระยะจาก centroid แกน x
        yi = y - yc  # ระยะจาก centroid แกน y

        # แรงแกนกลาง
        p_axial = P / n

        # แรงจากโมเมนต์ (ระวัง Ix=0 หรือ Iy=0)
        p_mx = (Mx * yi / Ix) if Ix > 1e-9 else 0.0
        p_my = (My * xi / Iy) if Iy > 1e-9 else 0.0

        pi = p_axial + p_mx + p_my
        forces.append(pi)

    return forces


# =============================================================================
# ฟังก์ชันที่ 4: ตรวจสอบกำลังรับน้ำหนักและแรงดึง
# =============================================================================
def check_capacity(forces: list[float], Q_allowable: float) -> dict:
    """
    ตรวจสอบ:
      1) แรงในเสาเข็มแต่ละต้น ≤ Q_allowable
      2) ไม่มีเสาเข็มรับแรงดึง (Pi < 0)

    Args:
        forces: แรงในแต่ละเสาเข็ม (kN)
        Q_allowable: กำลังรับน้ำหนักที่ยอมให้ (kN/ต้น)

    Returns:
        dict ที่มีข้อมูลผล: สถานะ, เสาเข็มเกิน, เสาเข็มดึง
    """
    results = {
        "all_ok": True,
        "has_tension": False,
        "exceeded_piles": [],   # เสาเข็มที่รับแรงเกิน Q_allowable
        "tension_piles": [],    # เสาเข็มที่รับแรงดึง (Pi < 0)
        "max_force": max(forces),
        "min_force": min(forces),
        "max_pile_index": forces.index(max(forces)) + 1,
        "min_pile_index": forces.index(min(forces)) + 1,
    }

    for i, pi in enumerate(forces, start=1):
        # ตรวจสอบแรงดึง
        if pi < 0:
            results["has_tension"] = True
            results["tension_piles"].append(i)
            results["all_ok"] = False

        # ตรวจสอบแรงเกินกำลัง
        if pi > Q_allowable:
            results["exceeded_piles"].append((i, pi))
            results["all_ok"] = False

    return results


# =============================================================================
# ฟังก์ชันที่ 5: แสดงผลลัพธ์แบบ formatted
# =============================================================================
def print_results(
    P, Mx, My,
    pile_coords, centroid,
    Ix, Iy,
    forces,
    Q_allowable,
    check,
    B, L, t,
    fc_prime, fy,
):
    """แสดงผลการคำนวณทั้งหมดในรูปแบบที่อ่านง่าย"""

    SEP = "=" * 68
    SEP2 = "-" * 68

    print()
    print(SEP)
    print("  ผลการคำนวณฐานรากเสาเข็มแบบเยื้องศูนย์  (มยผ.1106-64)")
    print(SEP)

    # ---- ข้อมูล Input ----
    print("\n[1] ข้อมูลนำเข้า (Input)")
    print(SEP2)
    print(f"  แรงอัด P             : {P:>10.2f}  kN")
    print(f"  โมเมนต์ Mx           : {Mx:>10.2f}  kN·m")
    print(f"  โมเมนต์ My           : {My:>10.2f}  kN·m")
    print(f"  จำนวนเสาเข็ม         : {len(pile_coords):>10d}  ต้น")
    print(f"  กำลังรับน้ำหนักเสาเข็ม: {Q_allowable:>10.2f}  kN/ต้น")
    print(f"  ขนาดฐานราก (B×L×t)  : {B:.2f} × {L:.2f} × {t:.2f}  ม.")
    print(f"  fc'                  : {fc_prime:>10.2f}  MPa")
    print(f"  fy                   : {fy:>10.2f}  MPa")

    # ---- Centroid ----
    print("\n[2] จุดศูนย์กลางกลุ่มเสาเข็ม (Centroid)")
    print(SEP2)
    xc, yc = centroid
    print(f"  xc = {xc:.4f}  ม.")
    print(f"  yc = {yc:.4f}  ม.")

    # ---- Moment of Inertia ----
    print("\n[3] Moment of Inertia ของกลุ่มเสาเข็ม")
    print(SEP2)
    print(f"  Ix = Σyi² = {Ix:.4f}  ม.²   (รอบแกน x)")
    print(f"  Iy = Σxi² = {Iy:.4f}  ม.²   (รอบแกน y)")

    # ---- แรงในเสาเข็มแต่ละต้น ----
    print("\n[4] แรงในเสาเข็มแต่ละต้น")
    print(SEP2)
    header = f"  {'ต้นที่':^6}  {'x (ม.)':^10}  {'y (ม.)':^10}  {'Pi (kN)':^12}  {'สถานะ':^10}"
    print(header)
    print("  " + "-" * 62)

    for i, ((x, y), pi) in enumerate(zip(pile_coords, forces), start=1):
        if pi < 0:
            status = "⚠ ดึง"
        elif pi > Q_allowable:
            status = "✗ เกิน"
        else:
            status = "✓ OK"
        print(f"  {i:^6}  {x:^10.3f}  {y:^10.3f}  {pi:^12.2f}  {status:^10}")

    # ---- สรุปผล ----
    print("\n[5] สรุปผลการตรวจสอบ")
    print(SEP2)
    print(f"  แรงสูงสุดในเสาเข็ม  : {check['max_force']:>10.2f} kN  (ต้นที่ {check['max_pile_index']})")
    print(f"  แรงต่ำสุดในเสาเข็ม  : {check['min_force']:>10.2f} kN  (ต้นที่ {check['min_pile_index']})")
    print(f"  กำลังรับน้ำหนักที่ยอม: {Q_allowable:>10.2f} kN/ต้น")

    print()
    # แจ้งเตือนแรงดึง
    if check["has_tension"]:
        print(f"  ⚠️  คำเตือน: เสาเข็มต้นที่ {check['tension_piles']} รับแรงดึง!")
        print("      ควรพิจารณาเพิ่มจำนวนเสาเข็มหรือปรับตำแหน่ง")

    # แจ้งเตือนแรงเกิน
    if check["exceeded_piles"]:
        for pile_no, force in check["exceeded_piles"]:
            print(f"  ✗  เสาเข็มต้นที่ {pile_no}: Pi = {force:.2f} kN > Q_allow = {Q_allowable:.2f} kN")

    # สถานะรวม
    print()
    if check["all_ok"]:
        print("  ┌──────────────────────────────────────┐")
        print("  │   ✅  ผลการออกแบบ: ผ่าน  (OK)        │")
        print("  └──────────────────────────────────────┘")
    else:
        print("  ┌──────────────────────────────────────┐")
        print("  │   ❌  ผลการออกแบบ: ไม่ผ่าน (NOT OK)  │")
        print("  └──────────────────────────────────────┘")

    print(SEP)
    print()


# =============================================================================
# ฟังก์ชันที่ 6: plot ตำแหน่งเสาเข็มและแรง
# =============================================================================
def plot_pile_foundation(pile_coords, forces, centroid, Q_allowable, B, L):
    """
    วาดผังตำแหน่งเสาเข็มพร้อมแสดงขนาดแรงในแต่ละต้น
    ใช้สีแสดงสถานะ: เขียว = OK, แดง = เกิน, ส้ม = ดึง
    """
    if not MATPLOTLIB_AVAILABLE:
        print("[INFO] ไม่สามารถแสดงกราฟได้เนื่องจากไม่มี matplotlib")
        return

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#ffffff")

    xc, yc = centroid

    # วาดแผ่นฐานราก (outline)
    rect = plt.Rectangle(
        (-B / 2, -L / 2), B, L,
        linewidth=2, edgecolor="#333333",
        facecolor="#e8edf5", alpha=0.5, zorder=1
    )
    ax.add_patch(rect)
    ax.text(0, -L / 2 - 0.15, f"ฐานราก {B:.1f}×{L:.1f} ม.",
            ha="center", fontsize=9, color="#555555")

    # วาดเสาเข็มแต่ละต้น
    for i, ((x, y), pi) in enumerate(zip(pile_coords, forces), start=1):
        # เลือกสีตามสถานะ
        if pi < 0:
            color = "#e67e22"   # ส้ม = แรงดึง
            edge  = "#d35400"
        elif pi > Q_allowable:
            color = "#e74c3c"   # แดง = เกิน
            edge  = "#c0392b"
        else:
            color = "#2ecc71"   # เขียว = OK
            edge  = "#27ae60"

        # วงกลมแทนเสาเข็ม
        circle = plt.Circle((x, y), 0.10,
                             color=color, ec=edge, linewidth=1.5, zorder=3)
        ax.add_patch(circle)

        # หมายเลขเสาเข็ม
        ax.text(x, y, str(i),
                ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=4)

        # แสดงค่าแรง
        ax.text(x, y + 0.18, f"{pi:.1f} kN",
                ha="center", va="bottom",
                fontsize=7.5, color="#222222", zorder=4)

    # วาด centroid
    ax.plot(xc, yc, marker="+", markersize=14, color="#c0392b",
            markeredgewidth=2.5, zorder=5, label=f"Centroid ({xc:.2f}, {yc:.2f})")

    # legend สี
    legend_elements = [
        mpatches.Patch(facecolor="#2ecc71", edgecolor="#27ae60", label="OK (Pi ≤ Q_allow)"),
        mpatches.Patch(facecolor="#e74c3c", edgecolor="#c0392b", label="เกิน Q_allow"),
        mpatches.Patch(facecolor="#e67e22", edgecolor="#d35400", label="แรงดึง (Pi < 0)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    # ตกแต่ง
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlabel("x (ม.)", fontsize=11)
    ax.set_ylabel("y (ม.)", fontsize=11)
    ax.set_title(
        "ผังตำแหน่งเสาเข็มและแรงที่กระทำ\n(Eccentric Pile Foundation — มยผ.1106-64)",
        fontsize=12, fontweight="bold"
    )

    margin = max(B, L) * 0.4
    ax.set_xlim(-B / 2 - margin, B / 2 + margin)
    ax.set_ylim(-L / 2 - margin, L / 2 + margin)

    plt.tight_layout()
    plt.show()


# =============================================================================
# ฟังก์ชันรับ Input จาก User
# =============================================================================
def get_user_input() -> dict:
    """รับค่า input จากผู้ใช้ผ่าน console"""
    print()
    print("=" * 68)
    print("  โปรแกรมออกแบบฐานรากเสาเข็มแบบเยื้องศูนย์  (มยผ.1106-64)")
    print("=" * 68)
    print("  กรอกข้อมูล (กด Enter เพื่อใช้ค่า default ในวงเล็บ)")
    print("-" * 68)

    def _input(prompt, default):
        val = input(f"  {prompt} [{default}]: ").strip()
        return type(default)(val) if val else default

    P  = _input("แรงอัด P (kN)", 1200.0)
    Mx = _input("โมเมนต์ Mx รอบแกน x (kN·m)", 150.0)
    My = _input("โมเมนต์ My รอบแกน y (kN·m)", 80.0)

    print()
    n = _input("จำนวนเสาเข็ม n (ต้น)", 4)

    pile_coords = []
    print(f"\n  กรอกตำแหน่ง (x, y) ของเสาเข็มทั้ง {n} ต้น (เมตร):")
    for i in range(1, n + 1):
        while True:
            try:
                raw = input(f"    เสาเข็มต้นที่ {i}  x, y = ").strip()
                parts = [p.strip() for p in raw.replace(",", " ").split()]
                x, y  = float(parts[0]), float(parts[1])
                pile_coords.append((x, y))
                break
            except (ValueError, IndexError):
                print("    ⚠ กรุณากรอกตัวเลข 2 ค่า เช่น  0.75  0.75")

    print()
    Q_allowable = _input("กำลังรับน้ำหนักของเสาเข็ม Q_allow (kN/ต้น)", 400.0)
    B  = _input("ความกว้างฐานราก B (ม.)", 2.5)
    L  = _input("ความยาวฐานราก L (ม.)", 2.5)
    t  = _input("ความหนาฐานราก t (ม.)", 0.6)
    fc = _input("กำลังคอนกรีต fc' (MPa)", 24.0)
    fy = _input("กำลังเหล็ก fy (MPa)", 390.0)

    return dict(P=P, Mx=Mx, My=My, pile_coords=pile_coords,
                Q_allowable=Q_allowable, B=B, L=L, t=t,
                fc_prime=fc, fy=fy)


# =============================================================================
# ฟังก์ชัน Main — รวมทุกขั้นตอน
# =============================================================================
def main(use_defaults: bool = False):
    """
    ฟังก์ชันหลัก ควบคุมการทำงานทั้งหมด

    Args:
        use_defaults: True = ใช้ค่าตัวอย่างโดยไม่ถามผู้ใช้
                      False = รับ input จาก console
    """

    if use_defaults:
        # --- ค่าตัวอย่าง (Demo) ---
        data = dict(
            P           = 1200.0,    # kN
            Mx          = 150.0,     # kN·m
            My          = 80.0,      # kN·m
            pile_coords = [          # เสาเข็ม 4 ต้น สมมาตร
                ( 0.75,  0.75),
                (-0.75,  0.75),
                (-0.75, -0.75),
                ( 0.75, -0.75),
            ],
            Q_allowable = 400.0,     # kN/ต้น
            B           = 2.5,       # ม.
            L           = 2.5,       # ม.
            t           = 0.6,       # ม.
            fc_prime    = 24.0,      # MPa
            fy          = 390.0,     # MPa
        )
        print("\n[INFO] ใช้ค่าตัวอย่าง (Demo mode)")
    else:
        data = get_user_input()

    # ดึงค่าออกมา
    P           = data["P"]
    Mx          = data["Mx"]
    My          = data["My"]
    pile_coords = data["pile_coords"]
    Q_allowable = data["Q_allowable"]
    B, L, t     = data["B"], data["L"], data["t"]
    fc_prime    = data["fc_prime"]
    fy          = data["fy"]

    # ---- คำนวณทีละขั้น ----

    # 1) centroid
    centroid = calculate_centroid(pile_coords)

    # 2) moment of inertia
    Ix, Iy = calculate_moment_of_inertia(pile_coords, centroid)

    # 3) แรงในเสาเข็ม
    forces = calculate_pile_forces(P, Mx, My, pile_coords, centroid, Ix, Iy)

    # 4) ตรวจสอบกำลัง
    check = check_capacity(forces, Q_allowable)

    # 5) แสดงผล
    print_results(
        P, Mx, My,
        pile_coords, centroid,
        Ix, Iy,
        forces,
        Q_allowable,
        check,
        B, L, t,
        fc_prime, fy,
    )

    # 6) plot (optional)
    if MATPLOTLIB_AVAILABLE:
        plot_pile_foundation(pile_coords, forces, centroid, Q_allowable, B, L)


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    # ส่ง argument "--demo" เพื่อข้ามการกรอก input
    # เช่น:  python pile_foundation_calculator.py --demo
    demo_mode = "--demo" in sys.argv
    main(use_defaults=demo_mode)
