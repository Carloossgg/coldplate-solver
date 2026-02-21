import numpy as np
import matplotlib
matplotlib.use('Agg') # 使用非交互式後端
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

def solve_thermal_adjoint(T_in=None, k_in=None, P_in=None, sensitivity_output_path=None):
    """
    熱傳導伴隨法分析程序。
    可選輸入:
    - T_in: 已知的溫度矩陣 (ny, nx)
    - k_in: 已知的熱傳導矩陣 (ny, nx)
    - P_in: 已知的熱源矩陣 (ny, nx)
    - sensitivity_output_path: 若提供，將靈敏度場寫入此 txt 檔 (tab 分隔)
    回傳: sens (ny, nx) 靈敏度場
    """
    # ==========================================
    # 1. 參數與網格設定 (Grid & Parameters)
    # ==========================================
    if T_in is not None and k_in is not None:
        print("Using external T and k matrices...")
        T = T_in
        k_field = k_in
        ny, nx = T.shape
        if P_in is not None:
            P = P_in
        else:
            P = np.zeros((ny, nx))
    else:
        print("Using default internal parameters...")
        nx, ny = 40, 40
        k_solid, k_fluid = 400.0, 0.6
        k_field = np.full((ny, nx), (k_solid + k_fluid) / 2.0)
        P = np.zeros((ny, nx))
        P[ny//4:3*ny//4, nx//4:3*nx//4] = 100.0

    n = nx * ny
    dx = 1.0 / nx
    P_vec = P.flatten()

    # ==========================================
    # 2. 構建熱傳導矩陣 K (為了 Adjoint Solve 仍需組裝)
    # ==========================================
    def assemble_K(k_vals):
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

    print("Step 1: Assembling K matrix...")
    K = assemble_K(k_field)
    
    # ==========================================
    # 3. 正向求解 (如果沒有傳入 T_in 則自行計算)
    # ==========================================
    if T_in is None:
        print("Step 2: Solving Forward problem (Temperature Field)...")
        T_vec = spsolve(K, P_vec)
        T = T_vec.reshape((ny, nx))
    else:
        print("Step 2: Using provided Temperature Field.")
    
    # ==========================================
    # 4. 伴隨求解 (Adjoint Solve)
    # ==========================================
    print("Step 3: Solving Adjoint problem (Lambda Field)...")
    dQ_dT = np.ones(n) / n
    lambda_vec = spsolve(K.T, dQ_dT)
    Lambda = lambda_vec.reshape((ny, nx))
    
    # ==========================================
    # 5. 靈敏度計算 (Sensitivity Calculation)
    # ==========================================
    print("Step 4: Calculating Sensitivities (dQ/dk)...")
    sens = np.zeros((ny, nx))
    for i in range(ny):
        for j in range(nx):
            grad_energy = 0
            neighbors = []
            if j > 0: neighbors.append((i, j-1))
            if j < nx-1: neighbors.append((i, j+1))
            if i > 0: neighbors.append((i-1, j))
            if i < ny-1: neighbors.append((i+1, j))
            
            for ni, nj in neighbors:
                k_i, k_j = k_field[i, j], k_field[ni, nj]
                dk_eff_dk_i = 2 * (k_j**2) / (k_i + k_j)**2
                temp_diff = T[i, j] - T[ni, nj]
                lambda_diff = Lambda[i, j] - Lambda[ni, nj]
                grad_energy += -(lambda_diff * (dk_eff_dk_i / dx**2) * temp_diff)
            
            boundary_count = 0
            if i == 0 or i == ny-1 or j == 0 or j == nx-1:
                if i == 0: boundary_count += 1
                if i == ny-1: boundary_count += 1
                if j == 0: boundary_count += 1
                if j == nx-1: boundary_count += 1
                grad_energy += -Lambda[i, j] * (2 * boundary_count / dx**2) * T[i, j]
            
            sens[i, j] = grad_energy

    # ==========================================
    # 6. 繪圖
    # ==========================================
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    
    # 圖 (1): 熱導率場 k
    im0 = axes[0].imshow(k_field, cmap='cividis', origin='lower')
    axes[0].set_title("Conductivity Field ($k$)")
    fig.colorbar(im0, ax=axes[0])
    
    # 圖 (2): 溫度場 T
    im1 = axes[1].imshow(T, cmap='hot', origin='lower')
    axes[1].set_title("Temperature Field ($T$)")
    fig.colorbar(im1, ax=axes[1])
    
    # 圖 (3): 伴隨場 lambda
    im2 = axes[2].imshow(Lambda, cmap='viridis', origin='lower')
    axes[2].set_title(r"Adjoint Field ($\lambda$)")
    fig.colorbar(im2, ax=axes[2])
    
    # 圖 (4): 靈敏度場 (Sensitivity Map)
    im3 = axes[3].imshow(sens, cmap='RdBu_r', origin='lower')
    axes[3].set_title("Sensitivity Map ($dQ/dk$)")
    fig.colorbar(im3, ax=axes[3])
    
    plt.tight_layout()
    plt.savefig('thermal_adjoint_results.png')
    print("Results saved to thermal_adjoint_results.png")

    # 可選：寫出靈敏度場到 txt
    if sensitivity_output_path:
        np.savetxt(sensitivity_output_path, sens, delimiter='\t', fmt='%.10e')
        print(f"Sensitivity field saved to {sensitivity_output_path}")

    return sens

if __name__ == "__main__":
    # 測試代碼：
    # solve_thermal_adjoint()
    pass
