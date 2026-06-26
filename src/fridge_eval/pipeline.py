import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from .config import *
from .image_io import fetch_image
from .detector import Detector


def compute_metrics(items):
    review_score = 0

    num_high_other = 0
    num_low_other = 0
    maidong_count = 0

    for name, conf, *_ in items:
        if name == "maidong":
            maidong_count += 1
        elif name == "other":
            if conf > OTHER_THRESH:
                num_high_other += 1
            else:
                num_low_other += 1

    if num_high_other > 0:
        review_score += 100 * num_high_other

    if num_low_other > 0:
        review_score += 40 * num_low_other

    if maidong_count < MAIDONG_MIN_COUNT:
        review_score += 20

    review_flag = review_score > 0

    return {
        "review_flag": review_flag,
        "review_score": review_score,
        "num_high_other": num_high_other,
        "num_low_other": num_low_other,
        "maidong_count": maidong_count
    }


def compute_layers_and_conf(items):
    try:
        ys = [i[2] for i in items]
        if len(ys) < 3:
            return None, None, 0.0

        import numpy as np
        from scipy.signal import find_peaks

        hist, bin_edges = np.histogram(ys, bins=20)
        peaks, _ = find_peaks(hist, distance=2)

        if len(peaks) == 0:
            return None, None, 0.0

        centers = [(bin_edges[p] + bin_edges[p+1]) / 2 for p in peaks]

        layers = [[] for _ in centers]
        for name, conf, cy, h in items:
            idx = min(range(len(centers)), key=lambda i: abs(cy - centers[i]))
            layers[idx].append((name, conf, cy, h))

        m_layer, other_layer = 0, 0
        for layer in layers:
            if any(n == "other" for n, _, _, _ in layer):
                other_layer += 1
            else:
                m_layer += 1

        return m_layer, other_layer, 1.0
    except:
        return None, None, 0.0


def run_pipeline(input_file):
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs('output', exist_ok=True)

    df = pd.read_excel(input_file)

    new_cols = [
        'review_flag','review_score',
        'num_high_other','num_low_other','maidong_count',
        'm_layer','other_layer','confidence','status'
    ]

    for c in new_cols:
        if c not in df.columns:
            df[c] = None

    detector = Detector(MODEL_PATH, CLASS_NAMES)

    BATCH_FETCH = 100

    for start in range(0, len(df), BATCH_FETCH):
        end = min(start + BATCH_FETCH, len(df))
        batch_df = df.iloc[start:end]

        with ThreadPoolExecutor(MAX_WORKERS) as exe:
            fetched = list(exe.map(
                lambda x: fetch_image(x[1], x[0], CACHE_DIR),
                batch_df.iterrows()
            ))

        images, idxs = [], []
        retry_download = []

        for idx, img in fetched:
            if img is not None:
                images.append(img)
                idxs.append(idx)
            else:
                retry_download.append(idx)

        for idx in retry_download:
            _, img = fetch_image(df.loc[idx], idx, CACHE_DIR)
            if img is not None:
                images.append(img)
                idxs.append(idx)
            else:
                df.at[idx, "review_flag"] = True
                df.at[idx, "status"] = "download_failed"

        for i in tqdm(range(0, len(images), BATCH_SIZE)):
            sub_imgs = images[i:i+BATCH_SIZE]
            sub_idxs = idxs[i:i+BATCH_SIZE]

            results = detector.infer_batch(sub_imgs)

            for j, r in enumerate(results):
                idx = sub_idxs[j]

                items = detector.analyze(r)

                if items is None:
                    r_retry = detector.infer_batch([sub_imgs[j]])[0]
                    items = detector.analyze(r_retry)

                    if items is None:
                        df.at[idx, "status"] = "detect_failed"
                        continue

                metrics = compute_metrics(items)
                for k, v in metrics.items():
                    df.at[idx, k] = v

                try:
                    m_layer, other_layer, conf = compute_layers_and_conf(items)
                    df.at[idx, "m_layer"] = m_layer
                    df.at[idx, "other_layer"] = other_layer
                    df.at[idx, "confidence"] = conf
                except:
                    df.at[idx, "confidence"] = 0.0

                df.at[idx, "status"] = "success"

                try:
                    os.remove(os.path.join(CACHE_DIR, f"{idx}.jpg"))
                except:
                    pass

            del results, sub_imgs

        df.to_excel(OUTPUT_FILE, index=False)

        del images, idxs, fetched

        print(f"processed {end}/{len(df)}")

    print("done")

