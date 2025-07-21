import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from simulate_mm import simulate_mm
import os

def generate_and_save_dataset(filename="heat_data.csv", num_workers=None):
    if num_workers is None:
        num_workers = os.cpu_count() or 1

    W_values = np.round(np.arange(0.2, 3.6 + 0.01, 0.01), 2)
    T_hot_values = np.arange(200, 401, 20)
    param_grid = [(W, T_hot) for T_hot in T_hot_values for W in W_values]

    print(f"Starting {len(param_grid)} simulations using {num_workers} CPU core(s)...")

    results = []
    with mp.Pool(processes=num_workers) as pool:
        for result in tqdm(pool.imap_unordered(simulate_mm, param_grid), total=len(param_grid)):
            results.append(result)

    df = pd.DataFrame(results, columns=['W', 'T_hot', 'Tc', 'Tc_std', 'Tm', 'Tm2'])
    print(df)
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

if __name__ == '__main__':
    generate_and_save_dataset()
