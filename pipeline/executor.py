from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from config import MAX_WORKERS

def run_parallel(df, detector, worker_func):
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(worker_func, idx, row, detector) for idx, row in df.iterrows()]
        for f in tqdm(as_completed(futures), total=len(futures)):
            results.append(f.result())
    return results
