import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from ultralytics import YOLO

from . import config
from .image_io import fetch_image
from .detector_with_boxes import Detector


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (area1 + area2 - inter)


def center_inside(box1, box2):
    cx = (box1[0] + box1[2]) / 2
    cy = (box1[1] + box1[3]) / 2
    return box2[0] <= cx <= box2[2] and box2[1] <= cy <= box2[3]


def match_box(box1, box2):
    return compute_iou(box1, box2) >= config.IOU_THRESH or center_inside(box1, box2)


def compute_metrics(items):
    review_score = 0
    num_high_other = 0
    num_low_other = 0
    maidong_count = 0

    for name, conf, *_ in items:
        if name == 'maidong':
            maidong_count += 1
        elif name == 'other':
            if conf > config.OTHER_THRESH:
                num_high_other += 1
            else:
                num_low_other += 1

    if num_high_other > 0:
        review_score += 100 * num_high_other
    if num_low_other > 0:
        review_score += 40 * num_low_other
    if maidong_count < config.MAIDONG_MIN_COUNT:
        review_score += 20

    return {
        'review_flag': review_score > 0,
        'review_score': review_score,
        'num_high_other': num_high_other,
        'num_low_other': num_low_other,
        'maidong_count': maidong_count
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

        for name, conf, cy, h, *_ in items:
            idx = min(range(len(centers)), key=lambda i: abs(cy - centers[i]))
            layers[idx].append((name, conf, cy, h))

        m_layer, other_layer = 0, 0
        for layer in layers:
            if any(x[0] == 'other' for x in layer):
                other_layer += 1
            else:
                m_layer += 1

        return m_layer, other_layer, 1.0
    except:
        return None, None, 0.0


def run_pipeline(input_file):
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    os.makedirs('output', exist_ok=True)

    df = pd.read_excel(input_file)

    new_cols = [
        'review_flag','review_score',
        'num_high_other','num_low_other','maidong_count',
        'juice_count','original_other_count','corrected_other_count',
        'm_layer','other_layer','confidence','status'
    ]

    for c in new_cols:
        if c not in df.columns:
            df[c] = None

    detector = Detector(config.MODEL_PATH, config.CLASS_NAMES)
    juice_model = YOLO(config.JUICE_MODEL_PATH)

    BATCH_FETCH = 100

    for start in range(0, len(df), BATCH_FETCH):
        end = min(start + BATCH_FETCH, len(df))
        batch_df = df.iloc[start:end]

        with ThreadPoolExecutor(config.MAX_WORKERS) as exe:
            fetched = list(exe.map(
                lambda x: fetch_image(x[1], x[0], config.CACHE_DIR),
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
            _, img = fetch_image(df.loc[idx], idx, config.CACHE_DIR)
            if img is not None:
                images.append(img)
                idxs.append(idx)
            else:
                df.at[idx, 'review_flag'] = True
                df.at[idx, 'status'] = 'download_failed'

        for i in tqdm(range(0, len(images), config.BATCH_SIZE)):
            sub_imgs = images[i:i+config.BATCH_SIZE]
            sub_idxs = idxs[i:i+config.BATCH_SIZE]

            results = detector.infer_batch(sub_imgs)

            for j, r in enumerate(results):
                idx = sub_idxs[j]

                items = detector.analyze(r)
                if items is None:
                    df.at[idx, 'status'] = 'detect_failed'
                    continue

                juice_result = juice_model([sub_imgs[j]], verbose=False)[0]
                juice_boxes = []
                if juice_result.boxes is not None:
                    for b in juice_result.boxes:
                        if float(b.conf[0]) >= config.JUICE_CONF_THRESH:
                            juice_boxes.append(b.xyxy[0].tolist())

                original_other_count = sum(1 for x in items if x[0] == 'other')
                juice_count = len(juice_boxes)

                filtered_items = []
                for item in items:
                    if item[0] != 'other':
                        filtered_items.append(item)
                        continue

                    other_box = item[4:8]
                    matched = any(match_box(other_box, jb) for jb in juice_boxes)

                    if not matched:
                        filtered_items.append(item)

                corrected_other_count = sum(1 for x in filtered_items if x[0] == 'other')

                metrics = compute_metrics(filtered_items)

                df.at[idx, 'juice_count'] = juice_count
                df.at[idx, 'original_other_count'] = original_other_count
                df.at[idx, 'corrected_other_count'] = corrected_other_count

                for k, v in metrics.items():
                    df.at[idx, k] = v

                m_layer, other_layer, conf = compute_layers_and_conf(filtered_items)
                df.at[idx, 'm_layer'] = m_layer
                df.at[idx, 'other_layer'] = other_layer
                df.at[idx, 'confidence'] = conf
                df.at[idx, 'status'] = 'success'

                try:
                    os.remove(os.path.join(config.CACHE_DIR, f"{idx}.jpg"))
                except:
                    pass

        df.to_excel(config.OUTPUT_FILE, index=False)
        print(f'processed {end}/{len(df)}')

    print('done')
