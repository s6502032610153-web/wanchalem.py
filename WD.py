# “””

# การคำนวณฐานรากเสาเข็มแบบเยื้องศูนย์ (Eccentric Pile Group Foundation)
มาตรฐาน มยผ.1106-64: มาตรฐานการออกแบบฐานรากและโครงสร้างใต้ดิน

# ผู้เขียน : วิศวกรโครงสร้าง
อ้างอิง  : มยผ.1106-64 หมวด 5 – ฐานรากเสาเข็ม

“””

from **future** import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────

# Data Classes

# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PilePosition:
“”“พิกัดเสาเข็มในระบบแกน XY (เมตร หรือ หน่วยเดียวกัน)”””
id: int
x: float   # ระยะตามแนวแกน X จากจุดอ้างอิง (m)
y: float   # ระยะตามแนวแกน Y จากจุดอ้างอิง (m)

```
def __repr__(self) -> str:
    return f"Pile-{self.id:02d}  x={self.x:+.3f} m  y={self.y:+.3f} m"
```

@dataclass
class FoundationInput:
“””
ข้อมูลนำเข้าสำหรับการคำนวณฐานรากเสาเข็มเยื้องศูนย์

```
Parameters
----------
P_kN        : แรงกดในแนวดิ่งที่ฐานราก (kN)  [บวก = กด]
Mx_kNm      : โมเมนต์รอบแกน X (kN-m)  [ตามกฎมือขวา]
My_kNm      : โมเมนต์รอบแกน Y (kN-m)  [ตามกฎมือขวา]
piles       : รายการตำแหน่งเสาเข็ม
Q_allow_kN  : กำลังรับน้ำหนักที่ยอมให้ต่อต้น (kN)
B_m         : ความกว้างฐานราก (m)
L_m         : ความยาวฐานราก (m)
t_m         : ความหนาฐานราก (m)
fc_prime_MPa: กำลังรับแรงอัดของคอนกรีต f'c (MPa)
fy_MPa      : กำลังครากของเหล็กเสริม fy (MPa)
"""
P_kN: float
Mx_kNm: float
My_kNm: float
piles: list[PilePosition]
Q_allow_kN: float
B_m: float
L_m: float
t_m: float
fc_prime_MPa: float
fy_MPa: float
```

@dataclass
class PileGroupGeometry:
“”“ผลการวิเคราะห์เรขาคณิตกลุ่มเสาเข็ม”””
n: int                          # จำนวนเสาเข็มทั้งหมด (ต้น)
cx: float                       # centroid X (m)
cy: float                       # centroid Y (m)
Ix: float                       # โมเมนต์ความเฉื่อยรอบแกน X ผ่าน centroid (m²)
Iy: float                       # โมเมนต์ความเฉื่อยรอบแกน Y ผ่าน centroid (m²)
piles_local: list[PilePosition]  # พิกัดเสาเข็มในระบบ local (เทียบ centroid)

@dataclass
class PileForceResult:
“”“แรงที่เสาเข็มแต่ละต้น”””
pile: PilePosition
P_avg: float        # แรงกดเฉลี่ย P/n (kN)
dP_Mx: float        # แรงจากโมเมนต์ Mx (kN)
dP_My: float        # แรงจากโมเมนต์ My (kN)
Pi: float           # แรงรวมในเสาเข็ม (kN)
is_tension: bool    # True ถ้าเป็นแรงดึง (Pi < 0)
ok: bool            # True ถ้า Pi ≤ Q_allow

@dataclass
class DesignSummary:
“”“สรุปผลการออกแบบ”””
geometry: PileGroupGeometry
results: list[PileForceResult]
P_max: float
P_min: float
Q_allow: float
has_tension: bool
all_capacity_ok: bool
overall_ok: bool

# ─────────────────────────────────────────────────────────────────────────────

# Core Functions

# ─────────────────────────────────────────────────────────────────────────────

def compute_pile_group_geometry(piles: list[PilePosition]) -> PileGroupGeometry:
“””
คำนวณ centroid และ moment of inertia ของกลุ่มเสาเข็ม

```
สูตร (มยผ.1106-64 ข้อ 5.3.2):
    centroid X = ΣXi / n
    centroid Y = ΣYi / n
    Ix = Σ(yi - cy)²
    Iy = Σ(xi - cx)²

Returns
-------
PileGroupGeometry
"""
n = len(piles)
if n == 0:
    raise ValueError("ต้องมีเสาเข็มอย่างน้อย 1 ต้น")

# ── Centroid ──────────────────────────────────────────────────────────────
cx = sum(p.x for p in piles) / n
cy = sum(p.y for p in piles) / n

# ── พิกัดในระบบ Local (เทียบ centroid) ──────────────────────────────────
piles_local = [
    PilePosition(id=p.id, x=p.x - cx, y=p.y - cy)
    for p in piles
]

# ── Moment of Inertia ─────────────────────────────────────────────────────
Ix = sum(p.y ** 2 for p in piles_local)   # รอบแกน X  (m²)
Iy = sum(p.x ** 2 for p in piles_local)   # รอบแกน Y  (m²)

return PileGroupGeometry(
    n=n, cx=cx, cy=cy,
    Ix=Ix, Iy=Iy,
    piles_local=piles_local
)
```

def compute_pile_forces(
inp: FoundationInput,
geo: PileGroupGeometry
) -> list[PileForceResult]:
“””
คำนวณแรงในเสาเข็มแต่ละต้น

```
สูตรหลัก (มยผ.1106-64 ข้อ 5.3.3):
    Pi = P/n ± (Mx · yi / Ix) ± (My · xi / Iy)

หมายเหตุ:
    - เครื่องหมาย ± ขึ้นกับทิศทางโมเมนต์และตำแหน่งเสาเข็ม
    - ถ้า Ix = 0 แสดงว่าเสาเข็มทั้งหมดอยู่บนแกน X → ไม่มีผลจาก Mx
    - ถ้า Iy = 0 แสดงว่าเสาเข็มทั้งหมดอยู่บนแกน Y → ไม่มีผลจาก My

Parameters
----------
inp : FoundationInput  – ข้อมูลนำเข้า
geo : PileGroupGeometry – เรขาคณิตกลุ่มเสาเข็ม

Returns
-------
list[PileForceResult]
"""
results: list[PileForceResult] = []

P_avg = inp.P_kN / geo.n   # แรงกดเฉลี่ยต่อต้น (kN)

for p_local, p_orig in zip(geo.piles_local, inp.piles):
    # แรงจากโมเมนต์ Mx (กระทำรอบแกน X → ผลต่อ yi)
    dP_Mx = (inp.Mx_kNm * p_local.y / geo.Ix) if geo.Ix > 1e-12 else 0.0

    # แรงจากโมเมนต์ My (กระทำรอบแกน Y → ผลต่อ xi)
    dP_My = (inp.My_kNm * p_local.x / geo.Iy) if geo.Iy > 1e-12 else 0.0

    Pi = P_avg + dP_Mx + dP_My

    results.append(PileForceResult(
        pile=p_orig,
        P_avg=P_avg,
        dP_Mx=dP_Mx,
        dP_My=dP_My,
        Pi=Pi,
        is_tension=(Pi < 0.0),
        ok=(Pi <= inp.Q_allow_kN)
    ))

return results
```

def check_design(inp: FoundationInput, results: list[PileForceResult],
geo: PileGroupGeometry) -> DesignSummary:
“””
ตรวจสอบและสรุปผลการออกแบบ

```
เงื่อนไข (มยผ.1106-64):
    1. Pi ≤ Q_allowable   ทุกต้น
    2. Pi ≥ 0             (ไม่มีแรงดึง)

Returns
-------
DesignSummary
"""
forces = [r.Pi for r in results]
P_max = max(forces)
P_min = min(forces)
has_tension = any(r.is_tension for r in results)
all_capacity_ok = all(r.ok for r in results)
overall_ok = all_capacity_ok and not has_tension

return DesignSummary(
    geometry=geo,
    results=results,
    P_max=P_max,
    P_min=P_min,
    Q_allow=inp.Q_allow_kN,
    has_tension=has_tension,
    all_capacity_ok=all_capacity_ok,
    overall_ok=overall_ok
)
```

def compute_eccentricity(inp: FoundationInput, geo: PileGroupGeometry) -> tuple[float, float]:
“””
คำนวณค่าความเยื้องศูนย์ (eccentricity)

```
ex = My / P   (ตามแกน X)
ey = Mx / P   (ตามแกน Y)

Returns
-------
(ex, ey) หน่วย เมตร
"""
ex = inp.My_kNm / inp.P_kN if abs(inp.P_kN) > 1e-12 else 0.0
ey = inp.Mx_kNm / inp.P_kN if abs(inp.P_kN) > 1e-12 else 0.0
return ex, ey
```

# ─────────────────────────────────────────────────────────────────────────────

# Report Printer

# ─────────────────────────────────────────────────────────────────────────────

def print_report(inp: FoundationInput, summary: DesignSummary) -> None:
“”“พิมพ์รายงานผลการคำนวณแบบละเอียด”””

```
SEP  = "=" * 72
SEP2 = "-" * 72
geo  = summary.geometry

print(SEP)
print("  ฐานรากเสาเข็มแบบเยื้องศูนย์  –  มยผ.1106-64")
print(SEP)

# ── 1. ข้อมูลนำเข้า ───────────────────────────────────────────────────────
print("\n[1] ข้อมูลนำเข้า (Loading & Material)")
print(SEP2)
print(f"  แรงกดในแนวดิ่ง          P   = {inp.P_kN:>10.2f} kN")
print(f"  โมเมนต์รอบแกน X         Mx  = {inp.Mx_kNm:>10.2f} kN-m")
print(f"  โมเมนต์รอบแกน Y         My  = {inp.My_kNm:>10.2f} kN-m")
print(f"  จำนวนเสาเข็ม            n   = {geo.n:>10d} ต้น")
print(f"  กำลังรับน้ำหนักที่ยอมให้  Qa  = {inp.Q_allow_kN:>10.2f} kN/ต้น")
print(f"  ขนาดฐานราก (B×L×t)          = {inp.B_m:.2f} × {inp.L_m:.2f} × {inp.t_m:.2f} m")
print(f"  กำลังคอนกรีต            f'c = {inp.fc_prime_MPa:>10.1f} MPa")
print(f"  กำลังเหล็กเสริม         fy  = {inp.fy_MPa:>10.1f} MPa")

# ── 2. ตำแหน่งเสาเข็ม ───────────────────────────────────────────────────
print(f"\n[2] ตำแหน่งเสาเข็ม (Global Coordinates)")
print(SEP2)
print(f"  {'ต้น':>4}   {'X (m)':>10}   {'Y (m)':>10}")
print(f"  {'-'*4}   {'-'*10}   {'-'*10}")
for p in inp.piles:
    print(f"  {p.id:>4}   {p.x:>10.3f}   {p.y:>10.3f}")

# ── 3. เรขาคณิตกลุ่มเสาเข็ม ─────────────────────────────────────────────
print(f"\n[3] เรขาคณิตกลุ่มเสาเข็ม (Pile Group Geometry)")
print(SEP2)
ex, ey = compute_eccentricity(inp, geo)
print(f"  Centroid X              cx  = {geo.cx:>10.4f} m")
print(f"  Centroid Y              cy  = {geo.cy:>10.4f} m")
print(f"  โมเมนต์ความเฉื่อย       Ix  = {geo.Ix:>10.4f} m²")
print(f"  โมเมนต์ความเฉื่อย       Iy  = {geo.Iy:>10.4f} m²")
print(f"  ความเยื้องศูนย์          ex  = {ex:>10.4f} m  (My/P)")
print(f"  ความเยื้องศูนย์          ey  = {ey:>10.4f} m  (Mx/P)")

# ── 4. พิกัดในระบบ Local ──────────────────────────────────────────────────
print(f"\n[4] พิกัดเสาเข็ม (Local – เทียบ Centroid)")
print(SEP2)
print(f"  {'ต้น':>4}   {'xi (m)':>10}   {'yi (m)':>10}   {'xi² (m²)':>10}   {'yi² (m²)':>10}")
print(f"  {'-'*4}   {'-'*10}   {'-'*10}   {'-'*10}   {'-'*10}")
for p in geo.piles_local:
    print(f"  {p.id:>4}   {p.x:>10.4f}   {p.y:>10.4f}   {p.x**2:>10.4f}   {p.y**2:>10.4f}")
print(f"  {'Σ':>4}   {'':>10}   {'':>10}   {geo.Iy:>10.4f}   {geo.Ix:>10.4f}")

# ── 5. แรงในเสาเข็ม ──────────────────────────────────────────────────────
print(f"\n[5] แรงในเสาเข็มแต่ละต้น (Pile Forces)")
print(SEP2)
print(f"  สูตร:  Pi = P/n  ±  (Mx·yi / Ix)  ±  (My·xi / Iy)")
print(f"         P/n = {inp.P_kN:.2f}/{geo.n} = {inp.P_kN/geo.n:.4f} kN")
print(SEP2)
hdr = (f"  {'ต้น':>4}  {'P/n':>9}  {'±Mx·yi/Ix':>11}  {'±My·xi/Iy':>11}"
       f"  {'Pi (kN)':>10}  {'Qa (kN)':>9}  {'ตรวจสอบ':>8}  {'หมายเหตุ':>8}")
print(hdr)
print(f"  {'-'*4}  {'-'*9}  {'-'*11}  {'-'*11}"
      f"  {'-'*10}  {'-'*9}  {'-'*8}  {'-'*8}")

for r in summary.results:
    status  = "✓ OK" if r.ok else "✗ NG"
    warning = "⚠ ดึง" if r.is_tension else ""
    print(
        f"  {r.pile.id:>4}  {r.P_avg:>9.2f}  {r.dP_Mx:>+11.2f}  {r.dP_My:>+11.2f}"
        f"  {r.Pi:>10.2f}  {inp.Q_allow_kN:>9.2f}  {status:>8}  {warning:>8}"
    )

# ── 6. สรุป ───────────────────────────────────────────────────────────────
print(f"\n[6] สรุปผลการออกแบบ (Design Summary)")
print(SEP2)
print(f"  แรงสูงสุดในเสาเข็ม   Pmax = {summary.P_max:>10.2f} kN")
print(f"  แรงต่ำสุดในเสาเข็ม   Pmin = {summary.P_min:>10.2f} kN")
print(f"  กำลังรับน้ำหนัก       Qa   = {summary.Q_allow:>10.2f} kN")
ratio = summary.P_max / summary.Q_allow * 100 if summary.Q_allow > 0 else 0
print(f"  Pmax / Qa                  = {ratio:>9.1f} %")

print()
if summary.has_tension:
    print("  ⚠  WARNING: มีเสาเข็มรับแรงดึง! ตรวจสอบการออกแบบและเพิ่มจำนวนเสาเข็ม")
    tension_piles = [r.pile.id for r in summary.results if r.is_tension]
    print(f"     เสาเข็มที่มีแรงดึง: {tension_piles}")

if not summary.all_capacity_ok:
    over_piles = [r.pile.id for r in summary.results if not r.ok]
    print(f"  ⚠  WARNING: เสาเข็มต้น {over_piles} รับน้ำหนักเกิน Q_allowable!")

status_str = "✓  PASS – ผ่านการตรวจสอบ" if summary.overall_ok else "✗  FAIL – ไม่ผ่านการตรวจสอบ"
print(f"\n  ผลการออกแบบโดยรวม:  {status_str}")
print(SEP)
```

# ─────────────────────────────────────────────────────────────────────────────

# Main Entry Point (Pipeline)

# ─────────────────────────────────────────────────────────────────────────────

def design_eccentric_pile_group(inp: FoundationInput) -> DesignSummary:
“””
ฟังก์ชันหลัก: คำนวณและตรวจสอบฐานรากเสาเข็มแบบเยื้องศูนย์

```
Pipeline:
    1. compute_pile_group_geometry  → centroid, Ix, Iy
    2. compute_pile_forces          → Pi แต่ละต้น
    3. check_design                 → สรุปผล OK/NG
    4. print_report                 → พิมพ์รายงาน

Parameters
----------
inp : FoundationInput

Returns
-------
DesignSummary
"""
geo     = compute_pile_group_geometry(inp.piles)
results = compute_pile_forces(inp, geo)
summary = check_design(inp, results, geo)
print_report(inp, summary)
return summary
```

# ─────────────────────────────────────────────────────────────────────────────

# Example Usage

# ─────────────────────────────────────────────────────────────────────────────

if **name** == “**main**”:
“””
ตัวอย่าง: ฐานรากเสาเข็ม 4 ต้น แบบ 2×2 พร้อมแรงเยื้องศูนย์
─────────────────────────────────────────────────────────────
แรงกระทำ:
P  = 2000 kN
Mx =  120 kN-m
My =   80 kN-m

```
ตำแหน่งเสาเข็ม (หน่วย m):
    (1) x=-0.75, y=-0.75
    (2) x=+0.75, y=-0.75
    (3) x=-0.75, y=+0.75
    (4) x=+0.75, y=+0.75

Q_allowable = 650 kN/ต้น
ขนาดฐานราก  = 2.5 × 2.5 × 0.80 m
f'c = 28 MPa,  fy = 392 MPa  (SD40)
"""

piles_example = [
    PilePosition(id=1, x=-0.75, y=-0.75),
    PilePosition(id=2, x=+0.75, y=-0.75),
    PilePosition(id=3, x=-0.75, y=+0.75),
    PilePosition(id=4, x=+0.75, y=+0.75),
]

inp_example = FoundationInput(
    P_kN         = 2000.0,
    Mx_kNm       =  120.0,
    My_kNm       =   80.0,
    piles        = piles_example,
    Q_allow_kN   =  650.0,
    B_m          =    2.5,
    L_m          =    2.5,
    t_m          =    0.80,
    fc_prime_MPa =   28.0,
    fy_MPa       =  392.0,
)

summary = design_eccentric_pile_group(inp_example)

# ── ตัวอย่างที่ 2: กรณีมีแรงดึง (tension) ─────────────────────────────
print("\n\n" + "=" * 72)
print("  ตัวอย่างที่ 2: กรณีมีแรงดึงเกิดขึ้น (Large Moment)")
print("=" * 72 + "\n")

piles_ex2 = [
    PilePosition(id=1, x=-1.0, y=-1.0),
    PilePosition(id=2, x=+1.0, y=-1.0),
    PilePosition(id=3, x=-1.0, y=+1.0),
    PilePosition(id=4, x=+1.0, y=+1.0),
]

inp_ex2 = FoundationInput(
    P_kN         =  800.0,
    Mx_kNm       =  600.0,   # โมเมนต์ขนาดใหญ่ → อาจเกิดแรงดึง
    My_kNm       =  200.0,
    piles        = piles_ex2,
    Q_allow_kN   =  500.0,
    B_m          =    3.0,
    L_m          =    3.0,
    t_m          =    0.90,
    fc_prime_MPa =   28.0,
    fy_MPa       =  392.0,
)

design_eccentric_pile_group(inp_ex2)
```