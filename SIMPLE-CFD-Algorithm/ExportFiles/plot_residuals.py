import matplotlib.pyplot as plt
import pandas as pd
import os
try:
    df = pd.read_csv('ExportFiles/residuals.txt', delim_whitespace=True)
    plt.figure(figsize=(10, 6))
    plt.semilogy(df['Iter'], df['MassResid'], label='Mass')
    plt.semilogy(df['Iter'], df['UResid'], label='U-Mom')
    plt.semilogy(df['Iter'], df['VResid'], label='V-Mom')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.title('Convergence History')
    plt.xlabel('Iteration')
    plt.ylabel('Residual')
    plt.savefig('ExportFiles/residuals.png')
    print('Plot saved to ExportFiles/residuals.png')
except Exception as e:
    print(f'Error plotting: {e}')
