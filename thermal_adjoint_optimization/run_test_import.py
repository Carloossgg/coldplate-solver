import numpy as np
import os
import sys

# 將當前目錄加入路徑以便匯入 thermal_adjoint
sys.path.append(os.getcwd())
from thermal_adjoint import solve_thermal_adjoint

def run_import_test():
    print("--- 正在從 txt 檔案載入矩陣 ---")
    
    # 載入數據
    try:
        T_loaded = np.loadtxt('T_input.txt')
        k_loaded = np.loadtxt('k_input.txt')
        
        # 嘗試載入 P 矩陣 (如果有的話)
        P_loaded = None
        if os.path.exists('P_input.txt'):
            P_loaded = np.loadtxt('P_input.txt')
            print("載入功率圖 P_input.txt")
        
        print(f"載入成功! 矩陣尺寸: {T_loaded.shape}")
        
        # 執行伴隨運算
        solve_thermal_adjoint(T_in=T_loaded, k_in=k_loaded, P_in=P_loaded)
        
        print("\n--- 測試完成! 結果已儲存至 thermal_adjoint_results.png ---")
        
    except Exception as e:
        print(f"測試失敗: {e}")

if __name__ == "__main__":
    run_import_test()
