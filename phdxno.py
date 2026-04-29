import pandas as pd

def calculate_eccentric_piles(P_service, Mx, My, pile_coords, Q_allow, footing_dims):
    """
    คำนวณแรงปฏิกิริยาในเสาเข็ม (Pile Reaction) สำหรับฐานรากเยื้องศูนย์
    ตามหลักการของ Elastic Method
    """
    B, L, t = footing_dims
    n = len(pile_coords)
    
    # 1. คำนวณน้ำหนักบรรทุกทั้งหมด (รวมน้ำหนักฐานราก - Concrete unit weight ~24 kN/m3)
    W_footing = B * L * t * 24.0
    P_total = P_service + W_footing
    
    # 2. หา Centroid ของกลุ่มเสาเข็ม (x_bar, y_bar)
    sum_x = sum(p[0] for p in pile_coords)
    sum_y = sum(p[1] for p in pile_coords)
    x_bar = sum_x / n
    y_bar = sum_y / n
    
    # 3. คำนวณ Inertia ของกลุ่มเสาเข็ม (I = sum(d^2)) และปรับตำแหน่งเข้าหา Centroid
    # ระยะห่างจาก Centroid: xi = x - x_bar
    sum_x2 = 0
    sum_y2 = 0
    adjusted_coords = []
    
    for x, y in pile_coords:
        xi = x - x_bar
        yi = y - y_bar
        sum_x2 += xi**2
        sum_y2 += yi**2
        adjusted_coords.append((xi, yi))
        
    # 4. คำนวณแรงในเสาเข็มแต่ละต้น
    # สูตร: Pi = (P/n) +/- (Mx * yi / sum_yi^2) +/- (My * xi / sum_xi^2)
    pile_forces = []
    for i, (xi, yi) in enumerate(adjusted_coords):
        # หมายเหตุ: Mx ทำให้เกิดแรงในแนวแกน y, My ทำให้เกิดแรงในแนวแกน x
        force = (P_total / n) + (Mx * yi / sum_y2) + (My * xi / sum_x2)
        pile_forces.append(round(force, 2))
        
    # 5. สรุปผล
    max_force = max(pile_forces)
    min_force = min(pile_forces)
    
    status = "✅ OK" if max_force <= Q_allow else "❌ NOT OK (Over Capacity)"
    tension_warning = "⚠️ WARNING: Tension detected in piles!" if min_force < 0 else "No tension."

    # แสดงผลลัพธ์
    print("-" * 30)
    print(f"--- รายงานการคำนวณฐานราก (มยผ.1106-64) ---")
    print(f"น้ำหนักฐานราก: {W_footing:.2 text=''} kN")
    print(f"น้ำหนักรวมลงเสาเข็ม (P_total): {P_total:.2f} kN")
    print("-" * 30)
    
    df = pd.DataFrame({
        'Pile No.': range(1, n + 1),
        'x (m)': [p[0] for p in pile_coords],
        'y (m)': [p[1] for p in pile_coords],
        'Force (kN)': pile_forces
    })
    print(df.to_string(index=False))
    
    print("-" * 30)
    print(f"Max Force: {max_force} kN | Allowable: {Q_allow} kN")
    print(f"Min Force: {min_force} kN")
    print(f"สถานะ: {status}")
    if min_force < 0:
        print(tension_warning)
    print("-" * 30)

# --- [Input Section] ---
# แรงจากเสาตอหม้อ (Service Load)
P = 1200.0   # kN
Mx = 150.0   # kN-m
My = 80.0    # kN-m

# ตำแหน่งเสาเข็ม (สมมติเสาเข็ม 4 ต้น วางเป็นรูปสี่เหลี่ยมจัตุรัสห่างกัน 1m)
# พิกัด (x, y) อ้างอิงจากมุมใดมุมหนึ่งหรือจุดศูนย์กลางก็ได้
piles = [
    (0.0, 0.0), (1.0, 0.0),
    (0.0, 1.0), (1.0, 1.0)
]

Q_allow = 450.0  # kN/ต้น (Safe Load)
footing = (2.0, 2.0, 0.6)  # B, L, t (เมตร)

# Run Calculation
calculate_eccentric_piles(P, Mx, My, piles, Q_allow, footing)
