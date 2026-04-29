"""
╔══════════════════════════════════════════════════════════════════════════╗
║   การออกแบบฐานรากเสาเข็มแบบเยื้องศูนย์ (Eccentric Pile Group)          ║
║   มาตรฐาน มยผ.1106-64 (EIT Standard for Foundation Design)              ║
║   Structural Engineering Module - Pile Foundation Analysis               ║
╚══════════════════════════════════════════════════════════════════════════╝

ทฤษฎีการคำนวณ:
-----------
สมการแรงในเสาเข็ม (Pile Load Equation):
    Pi = P/n ± (Mx·yi / Ix) ± (My·xi / Iy)

โดยที่:
    Pi  = แรงในเสาเข็มต้นที่ i (kN)
    P   = แรงกดตั้งฉาก (kN)
    n   = จำนวนเสาเข็มทั้งหมด
    Mx  = โมเมนต์รอบแกน X (kN-m)
    My  = โมเมนต์รอบแกน Y (kN-m)
    xi  = ระยะห่างจาก centroid ของกลุ่มเสาเข็มถึงเสาเข็มต้นที่ i ในแนว X
    yi  = ระยะห่างจาก centroid ของกลุ่มเสาเข็มถึงเสาเข็มต้นที่ i ในแนว Y
    Ix  = โมเมนต์อินเนอร์เชียของกลุ่มเสาเข็มรอบแกน X = Σ(yi²)
    Iy  = โมเมนต์อินเนอร์เชียของกลุ่มเสาเข็มรอบแกน Y = Σ(xi²)
"""

import math
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PilePosition:
    """ตำแหน่งเสาเข็มในระบบพิกัด (ก่อนปรับ centroid)"""
    id: int
    x: float  # m
    y: float  # m


@dataclass
class PileResult:
    """ผลการคำนวณแรงในเสาเข็มแต่ละต้น"""
    id: int
    x_orig: float          # ตำแหน่งเดิม X (m)
    y_orig: float          # ตำแหน่งเดิม Y (m)
    x_centroid: float      # ระยะจาก centroid ในแนว X (m)
    y_centroid: float      # ระยะจาก centroid ในแนว Y (m)
    P_axial: float         # แรงจากแรงกด P/n (kN)
    P_mx: float            # แรงจากโมเมนต์ Mx (kN)
    P_my: float            # แรงจากโมเมนต์ My (kN)
    P_total: float         # แรงรวม (kN)
    status: str            # OK / NOT OK
    has_tension: bool      # มีแรงดึงหรือไม่


@dataclass
class FoundationInput:
    """ข้อมูล Input สำหรับการออกแบบฐานราก"""
    # แรงและโมเมนต์ที่ฐานราก
    P: float               # แรงกดตั้งฉาก (kN)
    Mx: float              # โมเมนต์รอบแกน X (kN-m)
    My: float              # โมเมนต์รอบแกน Y (kN-m)

    # เสาเข็ม
    piles: list            # รายการ PilePosition
    Q_allowable: float     # ความสามารถรับแรงแต่ละต้น (kN)

    # ขนาดฐานราก
    B: float               # ความกว้าง (m)
    L: float               # ความยาว (m)
    t: float               # ความหนา (m)

    # วัสดุ
    fc_prime: float        # กำลังอัดคอนกรีต (MPa)
    fy: float              # กำลังคราก (MPa)

    # ค่า Default
    gamma_concrete: float = 24.0   # น้ำหนักคอนกรีต (kN/m³)


@dataclass
class FoundationResult:
    """ผลการคำนวณทั้งหมด"""
    centroid_x: float
    centroid_y: float
    Ix: float
    Iy: float
    pile_results: list
    P_max: float
    P_min: float
    Q_allowable: float
    overall_status: str
    has_any_tension: bool
    self_weight: float
    warnings: list = field(default_factory=list)
    info: list = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Core Functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_centroid(piles: list[PilePosition]) -> tuple[float, float]:
    """
    คำนวณ centroid ของกลุ่มเสาเข็ม
    
    สมการ:
        x̄ = Σ(xi) / n
        ȳ = Σ(yi) / n
    
    Parameters:
        piles: รายการตำแหน่งเสาเข็ม
    
    Returns:
        (centroid_x, centroid_y) หน่วย m
    """
    n = len(piles)
    if n == 0:
        raise ValueError("❌ ต้องมีเสาเข็มอย่างน้อย 1 ต้น")
    
    cx = sum(p.x for p in piles) / n
    cy = sum(p.y for p in piles) / n
    return cx, cy


def compute_moments_of_inertia(piles: list[PilePosition],
                                cx: float, cy: float) -> tuple[float, float]:
    """
    คำนวณโมเมนต์อินเนอร์เชียของกลุ่มเสาเข็ม
    
    สมการ:
        Ix = Σ(yi²)   (yi = y - ȳ)
        Iy = Σ(xi²)   (xi = x - x̄)
    
    Parameters:
        piles: รายการตำแหน่งเสาเข็ม
        cx, cy: centroid ของกลุ่มเสาเข็ม
    
    Returns:
        (Ix, Iy) หน่วย m²
    """
    Ix = sum((p.y - cy) ** 2 for p in piles)
    Iy = sum((p.x - cx) ** 2 for p in piles)
    return Ix, Iy


def compute_pile_loads(inp: FoundationInput) -> FoundationResult:
    """
    คำนวณแรงในเสาเข็มแต่ละต้น ตามสมการ:
        Pi = P/n ± (Mx·yi / Ix) ± (My·xi / Iy)
    
    Parameters:
        inp: FoundationInput object
    
    Returns:
        FoundationResult object พร้อมผลการคำนวณ
    """
    piles = inp.piles
    n = len(piles)
    
    if n == 0:
        raise ValueError("❌ ต้องมีเสาเข็มอย่างน้อย 1 ต้น")
    
    warnings = []
    info = []
    
    # ─── น้ำหนักตัวเองของฐานราก ───
    self_weight = inp.B * inp.L * inp.t * inp.gamma_concrete
    P_total_with_sw = inp.P + self_weight
    info.append(f"น้ำหนักฐานราก (Self Weight) = {self_weight:.2f} kN")
    info.append(f"แรงกดรวม (P + SW) = {P_total_with_sw:.2f} kN")
    
    # ─── Centroid ───
    cx, cy = compute_centroid(piles)
    
    # ─── Moments of Inertia ───
    Ix, Iy = compute_moments_of_inertia(piles, cx, cy)
    
    # ─── ตรวจสอบ Ix, Iy (ป้องกัน Division by Zero) ───
    if Ix == 0 and inp.Mx != 0:
        warnings.append(
            "⚠️  Ix = 0 (เสาเข็มอยู่แนวเดียวกันในแกน Y) "
            "แต่มีโมเมนต์ Mx → ไม่สามารถต้านทานได้!"
        )
    if Iy == 0 and inp.My != 0:
        warnings.append(
            "⚠️  Iy = 0 (เสาเข็มอยู่แนวเดียวกันในแกน X) "
            "แต่มีโมเมนต์ My → ไม่สามารถต้านทานได้!"
        )
    
    # ─── คำนวณแรงแต่ละต้น ───
    pile_results = []
    has_any_tension = False
    
    for pile in piles:
        xi = pile.x - cx   # ระยะจาก centroid ในแนว X
        yi = pile.y - cy   # ระยะจาก centroid ในแนว Y
        
        # แรงจากแรงกด P/n
        P_axial = P_total_with_sw / n
        
        # แรงจากโมเมนต์ Mx (รอบแกน X → ส่งผลในแนว Y)
        P_mx = (inp.Mx * yi / Ix) if Ix != 0 else 0.0
        
        # แรงจากโมเมนต์ My (รอบแกน Y → ส่งผลในแนว X)
        P_my = (inp.My * xi / Iy) if Iy != 0 else 0.0
        
        # แรงรวม
        Pi = P_axial + P_mx + P_my
        
        # ตรวจสอบแรงดึง
        has_tension = Pi < 0
        if has_tension:
            has_any_tension = True
        
        # ตรวจสอบกับ Q_allowable
        status = "OK ✓" if abs(Pi) <= inp.Q_allowable else "NOT OK ✗"
        
        pile_results.append(PileResult(
            id=pile.id,
            x_orig=pile.x,
            y_orig=pile.y,
            x_centroid=xi,
            y_centroid=yi,
            P_axial=P_axial,
            P_mx=P_mx,
            P_my=P_my,
            P_total=Pi,
            status=status,
            has_tension=has_tension,
        ))
    
    # ─── สรุปค่าสูงสุด/ต่ำสุด ───
    P_max = max(r.P_total for r in pile_results)
    P_min = min(r.P_total for r in pile_results)
    
    overall_status = (
        "OK ✓" if all(r.status == "OK ✓" for r in pile_results) else "NOT OK ✗"
    )
    
    if has_any_tension:
        warnings.append(
            "⚠️  มีเสาเข็มรับแรงดึง (Tension) → ควรพิจารณาจัดเรียงเสาเข็มใหม่ "
            "หรือเพิ่มจำนวนเสาเข็ม"
        )
    
    return FoundationResult(
        centroid_x=cx,
        centroid_y=cy,
        Ix=Ix,
        Iy=Iy,
        pile_results=pile_results,
        P_max=P_max,
        P_min=P_min,
        Q_allowable=inp.Q_allowable,
        overall_status=overall_status,
        has_any_tension=has_any_tension,
        self_weight=self_weight,
        warnings=warnings,
        info=info,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Report Function
# ─────────────────────────────────────────────────────────────────────────────

def print_report(inp: FoundationInput, res: FoundationResult) -> None:
    """พิมพ์รายงานผลการคำนวณแบบละเอียด"""
    
    W = 72  # ความกว้างหน้ากระดาษ
    
    def line(char="─", w=W):
        print(char * w)
    
    def header(text, char="═"):
        print(char * W)
        print(f"  {text}")
        print(char * W)
    
    def section(text):
        print()
        print(f"┌{'─' * (W-2)}┐")
        print(f"│  {text:<{W-4}}│")
        print(f"└{'─' * (W-2)}┘")
    
    # ═══════════════════════════════════════════════════════════
    header("การออกแบบฐานรากเสาเข็มแบบเยื้องศูนย์ (Eccentric Pile Group)")
    print("  มาตรฐาน มยผ.1106-64 | EIT Standard for Foundation Design")
    line("═")
    
    # ─── Input Summary ───
    section("1. ข้อมูลนำเข้า (Input Summary)")
    print(f"  แรงและโมเมนต์ที่ฐานราก:")
    print(f"    P   = {inp.P:>10.2f} kN     (แรงกดตั้งฉาก)")
    print(f"    Mx  = {inp.Mx:>10.2f} kN-m  (โมเมนต์รอบแกน X)")
    print(f"    My  = {inp.My:>10.2f} kN-m  (โมเมนต์รอบแกน Y)")
    print()
    print(f"  ขนาดฐานราก:")
    print(f"    B = {inp.B:.2f} m | L = {inp.L:.2f} m | t = {inp.t:.2f} m")
    print(f"    พื้นที่ = {inp.B*inp.L:.2f} m²")
    print()
    print(f"  วัสดุ:")
    print(f"    f'c = {inp.fc_prime:.1f} MPa | fy = {inp.fy:.1f} MPa")
    print()
    print(f"  เสาเข็ม:")
    print(f"    จำนวน n = {len(inp.piles)} ต้น")
    print(f"    Q_allowable = {inp.Q_allowable:.2f} kN/ต้น")
    
    # ─── Pile Positions ───
    section("2. ตำแหน่งเสาเข็ม (Pile Positions)")
    print(f"  {'ต้นที่':>4}  {'X (m)':>8}  {'Y (m)':>8}")
    line()
    for p in inp.piles:
        print(f"  {p.id:>4}  {p.x:>8.3f}  {p.y:>8.3f}")
    
    # ─── Geometry ───
    section("3. เรขาคณิตกลุ่มเสาเข็ม (Pile Group Geometry)")
    print(f"  Centroid ของกลุ่มเสาเข็ม:")
    print(f"    x̄ = {res.centroid_x:.4f} m")
    print(f"    ȳ = {res.centroid_y:.4f} m")
    print()
    print(f"  โมเมนต์อินเนอร์เชีย:")
    print(f"    Ix = Σ(yi²) = {res.Ix:.4f} m²")
    print(f"    Iy = Σ(xi²) = {res.Iy:.4f} m²")
    
    # ─── Info ───
    if res.info:
        print()
        print("  ข้อมูลเพิ่มเติม:")
        for msg in res.info:
            print(f"    ℹ  {msg}")
    
    # ─── Pile Load Results ───
    section("4. แรงในเสาเข็มแต่ละต้น (Pile Load Results)")
    
    col_w = [4, 7, 7, 7, 7, 9, 9, 9, 10, 9]
    headers = ["ต้น", "X(m)", "Y(m)", "xi(m)", "yi(m)",
               "P/n(kN)", "Pmx(kN)", "Pmy(kN)", "Pi(kN)", "สถานะ"]
    
    # Header row
    header_str = "  " + "  ".join(f"{h:>{w}}" for h, w in zip(headers, col_w))
    print(header_str)
    line()
    
    for r in res.pile_results:
        tension_mark = " (T)" if r.has_tension else "    "
        row = "  " + "  ".join([
            f"{r.id:>{col_w[0]}}",
            f"{r.x_orig:>{col_w[1]}.3f}",
            f"{r.y_orig:>{col_w[2]}.3f}",
            f"{r.x_centroid:>{col_w[3]}.3f}",
            f"{r.y_centroid:>{col_w[4]}.3f}",
            f"{r.P_axial:>{col_w[5]}.2f}",
            f"{r.P_mx:>{col_w[6]}.2f}",
            f"{r.P_my:>{col_w[7]}.2f}",
            f"{r.P_total:>{col_w[8]}.2f}{tension_mark}",
            f"{r.status:>{col_w[9]}}",
        ])
        print(row)
    
    # ─── Summary ───
    section("5. สรุปผลการตรวจสอบ (Design Check Summary)")
    print(f"  แรงสูงสุดในเสาเข็ม  P_max = {res.P_max:>10.2f} kN")
    print(f"  แรงต่ำสุดในเสาเข็ม  P_min = {res.P_min:>10.2f} kN")
    print(f"  ความสามารถรับแรง   Q_all = {res.Q_allowable:>10.2f} kN/ต้น")
    print()
    
    # Check result box
    status_icon = "✓" if res.overall_status == "OK ✓" else "✗"
    status_label = "ผ่าน" if res.overall_status == "OK ✓" else "ไม่ผ่าน"
    print(f"  ┌{'─'*40}┐")
    print(f"  │  ผลการตรวจสอบ: {res.overall_status}  ({status_label})  {'':>10}│")
    print(f"  │  P_max / Q_all = {res.P_max/res.Q_allowable:.3f}            {'':>7}│")
    print(f"  └{'─'*40}┘")
    
    # ─── Warnings ───
    if res.warnings:
        print()
        print("  ⚠  คำเตือน (Warnings):")
        for w in res.warnings:
            print(f"     {w}")
    
    line("═")
    print("  End of Report — มยผ.1106-64 Eccentric Pile Group Analysis")
    line("═")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Wrapper
# ─────────────────────────────────────────────────────────────────────────────

def design_pile_foundation(
    P: float,
    Mx: float,
    My: float,
    pile_coords: list[tuple[float, float]],
    Q_allowable: float,
    B: float,
    L: float,
    t: float,
    fc_prime: float,
    fy: float,
    print_output: bool = True,
) -> FoundationResult:
    """
    ฟังก์ชันหลักสำหรับออกแบบฐานรากเสาเข็มแบบเยื้องศูนย์
    
    Parameters:
    -----------
    P           : แรงกดตั้งฉากที่ฐานราก (kN)
    Mx          : โมเมนต์รอบแกน X ที่ฐานราก (kN-m)
    My          : โมเมนต์รอบแกน Y ที่ฐานราก (kN-m)
    pile_coords : รายการ (x, y) ตำแหน่งเสาเข็ม เช่น [(0,0), (1.5,0), ...]
    Q_allowable : ความสามารถรับแรงของเสาเข็มแต่ละต้น (kN)
    B           : ความกว้างฐานราก (m)
    L           : ความยาวฐานราก (m)
    t           : ความหนาฐานราก (m)
    fc_prime    : กำลังอัดคอนกรีต f'c (MPa)
    fy          : กำลังครากเหล็ก fy (MPa)
    print_output: พิมพ์รายงาน (default: True)
    
    Returns:
    --------
    FoundationResult object
    
    Example:
    --------
    >>> result = design_pile_foundation(
    ...     P=2500, Mx=200, My=150,
    ...     pile_coords=[(0,0),(1.5,0),(3,0),(0,1.5),(1.5,1.5),(3,1.5)],
    ...     Q_allowable=600, B=4.5, L=3.0, t=1.0, fc_prime=28, fy=392
    ... )
    """
    piles = [PilePosition(id=i+1, x=c[0], y=c[1])
             for i, c in enumerate(pile_coords)]
    
    inp = FoundationInput(
        P=P, Mx=Mx, My=My,
        piles=piles,
        Q_allowable=Q_allowable,
        B=B, L=L, t=t,
        fc_prime=fc_prime,
        fy=fy,
    )
    
    result = compute_pile_loads(inp)
    
    if print_output:
        print_report(inp, result)
    
    return result


# ─────────────────────────────────────────────────────────────────────────────
# ตัวอย่างการใช้งาน (Example Usage)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    
    print("\n" + "═"*72)
    print("  ตัวอย่างที่ 1: ฐานรากเสาเข็ม 6 ต้น แบบ 2×3 (2 แถว × 3 คอลัมน์)")
    print("═"*72)
    
    result_1 = design_pile_foundation(
        P           = 2_500,   # kN
        Mx          = 200,     # kN-m  (โมเมนต์รอบแกน X)
        My          = 150,     # kN-m  (โมเมนต์รอบแกน Y)
        pile_coords = [
            (0.0, 0.0),        # เสาเข็มต้นที่ 1
            (1.5, 0.0),        # เสาเข็มต้นที่ 2
            (3.0, 0.0),        # เสาเข็มต้นที่ 3
            (0.0, 1.5),        # เสาเข็มต้นที่ 4
            (1.5, 1.5),        # เสาเข็มต้นที่ 5
            (3.0, 1.5),        # เสาเข็มต้นที่ 6
        ],
        Q_allowable = 600,     # kN/ต้น
        B           = 4.5,     # m
        L           = 3.0,     # m
        t           = 1.0,     # m
        fc_prime    = 28,      # MPa  (คอนกรีต)
        fy          = 392,     # MPa  (SD40)
    )

    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n" + "═"*72)
    print("  ตัวอย่างที่ 2: ฐานรากเสาเข็ม 4 ต้น มีแรงดึง (Tension Warning)")
    print("═"*72)
    
    result_2 = design_pile_foundation(
        P           = 800,     # kN  (แรงน้อย แต่โมเมนต์มาก)
        Mx          = 500,     # kN-m  (โมเมนต์สูงมาก)
        My          = 300,     # kN-m
        pile_coords = [
            (0.0, 0.0),
            (2.0, 0.0),
            (0.0, 2.0),
            (2.0, 2.0),
        ],
        Q_allowable = 500,     # kN/ต้น
        B           = 3.5,     # m
        L           = 3.5,     # m
        t           = 0.9,     # m
        fc_prime    = 24,      # MPa
        fy          = 392,     # MPa
    )

    # ─────────────────────────────────────────────────────────────────────────
    # การเรียกใช้งานแบบกำหนดเอง (Custom Usage Example)
    # ─────────────────────────────────────────────────────────────────────────
    
    # เข้าถึงผลลัพธ์แบบ Programmatic
    print("\n  📊 การเข้าถึงผลลัพธ์แบบ Programmatic:")
    print(f"     P_max = {result_1.P_max:.2f} kN")
    print(f"     P_min = {result_1.P_min:.2f} kN")
    print(f"     สถานะ: {result_1.overall_status}")
    print(f"     มีแรงดึง: {'ใช่' if result_1.has_any_tension else 'ไม่มี'}")
    
    for r in result_1.pile_results:
        print(f"     เสาเข็ม {r.id}: Pi = {r.P_total:.2f} kN  [{r.status}]")
