import os
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

from .config import *
from .image_io import fetch_image
from .detector import Detector


def compute_metrics(layers):
    m_layer = 0
    other_layer = 0

    review_score = 0
    num_high_other = 0
    num_low_other = 0
    num_low_maidong = 0

    for layer in layers:
        maidong_count = sum(1 for _, n, _ in layer if n == 'maidong')

        other_high = sum(1 for _, n, c in layer if n == 'other' and c > OTHER_THRESH)
        other_low = sum(1 for _, n, c in layer if n == 'other' and c <= OTHER_THRESH)

        if other_high > 0 or other_low > 0:
            other_layer += 1
        else:
            m_layer += 1

        if other_high > 0:
            num_high_other += 1
            review_score += 100

        if other_low > 0:
            num_low_other += 1
            review_score += 40

        if maidong_count < MAIDONG_MIN_COUNT:
            num_low_maidong += 1
            review_score += 20

    review_flag = review_score > 0

    return {
        'm_layer': m_layer,
        'other_layer': other_layer,
        'review_flag': review_flag,
        'review_score': review_score,
        'num_high_other': num_high_other,
        'num_low_other': num_low_other,
        'num_low_maidong_layers': num_low_maidong
    }


def process_chunk(df_chunk):
    detector = Detector(MODEL_PATH, CLASS_NAMES)

    results = []

    for idx, row in df_chunk.iterrows():
        _, img = fetch_image(row, idx, CACHE_DIR)

        if img is None:
            results.append((idx, {'review_flag': True}))
            continue

        res = detector.infer_batch([img])[0]

        layers = detector.analyze(res)
        if layers is None:
            results.append((idx, {}))
            continue

        metrics = compute_metrics(layers)
        results.append((idx, metrics))

        try:
            os.remove(os.path.join(CACHE_DIR, f"{idx}.jpg"))
        except:
            pass

    return results


def run_pipeline(input_file):
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs('output', exist_ok=True)

    df = pd.read_excel(input_file)

    new_cols = [
        'review_flag','review_score',
        'num_high_other','num_low_other',
        'num_low_maidong_layers','m_layer','other_layer'
    ]

    for c in new_cols:
        if c not in df.columns:
            df[c] = None

    num_proc = 4

    chunk_size = len(df) // num_proc + 1

    chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]

    with Pool(num_proc) as pool:
        results = list(tqdm(pool.imap(process_chunk, chunks), total=len(chunks)))

    for chunk_result in results:
        for idx, metrics in chunk_result:
            for k, v in metrics.items():
                df.at[idx, k] = v

    df.to_excel(OUTPUT_FILE, index=False)

    print('Done - multiprocessing version')
