import numpy as np
import os
import sys

# 將目錄加入路徑以便使用熱傳導組裝
sys.path.append(os.getcwd())
from thermal_adjoint import solve_thermal_adjoint
from scipy.sparse.linalg import spsolve

def generate_cpu_heatsink_data():
    # 提高解析度以模擬更細緻的散熱片
    nx, ny = 80, 80
    dx = 1.0 / nx
    
    # 1. 定義熱導率 k (Heat Sink Geometry)
    k_solid = 400.0 # 鋁/銅
    k_fluid = 0.026 # 空氣
    k_data = np.full((ny, nx), k_fluid)
    
    # 底座 (Base Plate): 底部 10% 區域
    base_height = ny // 10
    k_data[:base_height, :] = k_solid
    
    # 散熱鰭片 (Fins): 每隔一段距離垂直向上
    fin_width = 2
    fin_spacing = 8
    fin_top = int(ny * 0.8)
    for j in range(fin_spacing // 2, nx, fin_spacing):
        k_data[base_height:fin_top, j:j+fin_width] = k_solid
        
    # 2. 定義熱源 P (Random Square Heat Sources)
    # 模擬多個隨機分佈在底部的正方形熱源
    np.random.seed(42) # 固定隨機種子以利重複測試
    P_data = np.zeros((ny, nx))
    num_spots = 10
    spot_size = 4
    
    for _ in range(num_spots):
        # 限制隨機熱點位在底座區域 (底部 10%)
        cx = np.random.randint(spot_size, nx - spot_size)
        cy = np.random.randint(0, base_height - spot_size//2)
        
        # 繪製正方形熱點
        y_min, y_max = max(0, cy - spot_size//2), min(ny, cy + spot_size//2)
        x_min, x_max = max(0, cx - spot_size//2), min(nx, cx + spot_size//2)
        P_data[y_min:y_max, x_min:x_max] = 10000.0
    
    # 3. 獲取真實的溫度場 T (透過內置組裝器求解，確保測試資料是物理正確的)
    print("正在計算 CPU 散熱片的參考溫度場 (Forward Solve)...")
    
    # 這裡我們手動利用 thermal_adjoint.py 裡的邏輯來解一次 T
    # 為了簡化，我們直接調用 K 組裝邏輯 (這部分邏輯與 thermal_adjoint.py 相同)
    from scipy.sparse import csr_matrix
    def temp_assemble_K(k_vals, nx, ny, dx):
        n = nx * ny
        rows, cols, vals = [], [], []
        for i in range(ny):
            for j in range(nx):
                idx = i * nx + j
                diag_val = 0
                neighbors = []
                if j > 0: neighbors.append((i, j-1, idx-1))
                if j < nx-1: neighbors.append((i, j+1, idx+1))
                if i > 0: neighbors.append((i-1, j, idx-nx))
                if i < ny-1: neighbors.append((i+1, j, idx+nx))
                for ni, nj, n_idx in neighbors:
                    k_eff = 2 * k_vals[i, j] * k_vals[ni, nj] / (k_vals[i, j] + k_vals[ni, nj])
                    coeff = k_eff / dx**2
                    rows.append(idx); cols.append(n_idx); vals.append(-coeff)
                    diag_val += coeff
                boundary_count = 0
                if j == 0: boundary_count += 1
                if j == nx-1: boundary_count += 1
                if i == 0: boundary_count += 1
                if i == ny-1: boundary_count += 1
                if boundary_count > 0:
                    diag_val += 2 * boundary_count * k_vals[i, j] / dx**2
                rows.append(idx); cols.append(idx); vals.append(diag_val)
        return csr_matrix((vals, (rows, cols)), shape=(n, n))

    K = temp_assemble_K(k_data, nx, ny, dx)
    T_vec = spsolve(K, P_data.flatten())
    T_data = T_vec.reshape((ny, nx))
    
    # 4. 儲存數據
    np.savetxt('T_input.txt', T_data)
    np.savetxt('k_input.txt', k_data)
    np.savetxt('P_input.txt', P_data)
    
    print(f"成功建立 CPU 散熱片模擬數據 (解析度 {nx}x{ny})")
    print("- T_input.txt (由物理算出的溫度)")
    print("- k_input.txt (散熱片幾何)")
    print("- P_input.txt (CPU 熱源)")

if __name__ == "__main__":
    generate_cpu_heatsink_data()