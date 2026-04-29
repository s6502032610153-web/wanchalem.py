import pandas as pd

def calculate_eccentric_piles(P_service, Mx, My, pile_coords, Q_allow, footing_dims):
    """
    คำนวณแรงปฏิกิริยาในเสาเข็ม (Pile Reaction) ตามมาตรฐาน มยผ.1106-64
    แก้ไข Error ในส่วน f-string formatting
    """
    B, L, t = footing_dims
    n = len(pile_coords)
    
    # 1. คำนวณน้ำหนักฐานราก (Self-weight) 
    # คอนกรีตเสริมเหล็กตามมาตรฐาน มยผ. คิดที่ 24 kN/m3
    W_footing = B * L * t * 24.0
    P_total = P_service + W_footing
    
    # 2. หา Centroid ของกลุ่มเสาเข็ม
    sum_x = sum(p[0] for p in pile_coords)
    sum_y = sum(p[1] for p in pile_coords)
    x_bar = sum_x / n
    y_bar = sum_y / n
    
    # 3. คำนวณค่า I (sum of d^2) รอบแกน Centroid
    sum_x2 = 0
    sum_y2 = 0
    adjusted_coords = []
    
    for x, y in pile_coords:
        xi = x - x_bar
        yi = y - y_bar
        sum_x2 += xi**2
        sum_y2 += yi**2
        adjusted_coords.append((xi, yi))
        
    # 4. คำนวณแรงในเสาเข็มแต่ละต้น (Elastic Method)
    # Pi = (P/n) + (Mx*yi / sum_y2) + (My*xi / sum_x2)
    pile_forces = []
    for i, (xi, yi) in enumerate(adjusted_coords):
        # ปรับเครื่องหมายตามทิศทาง Moment (ในที่นี้ใช้บวกเพิ่มแรง)
        force = (P_total / n) + (Mx * yi / sum_y2) + (My * xi / sum_x2)
        pile_forces.append(round(force, 2))
        
    # 5. สรุปผล
    max_force = max(pile_forces)
    min_force = min(pile_forces)
    
    status = "✅ OK" if max_force <= Q_allow else "❌ NOT OK (Over Capacity)"
    
    # Output แสดงผล
    print("-" * 40)
    print("--- รายงานผลการคำนวณฐานรากเสาเข็ม ---")
    print(f"น้ำหนักฐานราก (Self-weight): {W_footing:.2f} kN")
    print(f"น้ำหนักรวม (Vertical Load): {P_total:.2f} kN")
    print("-" * 40)
    
    df = pd.DataFrame({
        'Pile No.': range(1, n + 1),
        'x-coord (m)': [p[0] for p in pile_coords],
        'y-coord (m)': [p[1] for p in pile_coords],
        'Reaction (kN)': pile_forces
    })
    print(df.to_string(index=False))
    
    print("-" * 40)
    print(f"Max Force: {max_force:.2f} kN / Allowable: {Q_allow:.2f} kN")
    print(f"Min Force: {min_force:.2f} kN")
    print(f"ผลการตรวจสอบ: {status}")
    
    if min_force < 0:
        print(f"⚠️ คำเตือน: พบแรงดึงในเสาเข็ม {min_force:.2f} kN")
    print("-" * 40)

# --- ส่วนของการป้อนข้อมูล (Input) ---
P = 1200.0   
Mx = 150.0   
My = 80.0    
Q_allow = 450.0  
footing = (2.0, 2.0, 0.6) # B, L, t

# พิกัดเสาเข็ม (x, y)
piles = [
    (0.0, 0.0), (1.0, 0.0),
    (0.0, 1.0), (1.0, 1.0)
]

# เรียกใช้งานฟังก์ชัน
if __name__ == "__main__":
    calculate_eccentric_piles(P, Mx, My, piles, Q_allow, footing)
