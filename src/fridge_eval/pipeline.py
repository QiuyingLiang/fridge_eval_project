import os,pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from .config import *
from .image_io import fetch_image
from .detector import Detector

def compute_metrics(layers):
    m_layer=0
    other_layer=0
    review_score=0
    num_high_other=0
    num_low_other=0
    num_low_maidong=0

    for layer in layers:
        maidong_count=sum(1 for _,n,_ in layer if n=='maidong')
        other_high=sum(1 for _,n,c in layer if n=='other' and c>OTHER_THRESH)
        other_low=sum(1 for _,n,c in layer if n=='other' and c<=OTHER_THRESH)

        if other_high>0 or other_low>0:
            other_layer+=1
        else:
            m_layer+=1

        if other_high>0:
            num_high_other+=1
            review_score+=100
        if other_low>0:
            num_low_other+=1
            review_score+=40
        if maidong_count<MAIDONG_MIN_COUNT:
            num_low_maidong+=1
            review_score+=20

    review_flag=review_score>0

    return {
        'm_layer':m_layer,
        'other_layer':other_layer,
        'review_flag':review_flag,
        'review_score':review_score,
        'num_high_other':num_high_other,
        'num_low_other':num_low_other,
        'num_low_maidong_layers':num_low_maidong
    }


def run_pipeline(input_file):
    os.makedirs(CACHE_DIR,exist_ok=True)
    os.makedirs('output',exist_ok=True)

    df=pd.read_excel(input_file)

    new_cols=['review_flag','review_score','num_high_other','num_low_other','num_low_maidong_layers','m_layer','other_layer']
    for c in new_cols:
        if c not in df.columns:
            df[c]=None

    detector=Detector(MODEL_PATH,CLASS_NAMES)

    with ThreadPoolExecutor(MAX_WORKERS) as exe:
        fetched=list(exe.map(lambda x:fetch_image(x[1],x[0],CACHE_DIR),df.iterrows()))

    images,idxs=[],[]
    for idx,img in fetched:
        if img is not None:
            images.append(img)
            idxs.append(idx)
        else:
            df.at[idx,'review_flag']=True

    for i in tqdm(range(0,len(images),BATCH_SIZE)):
        res=detector.infer_batch(images[i:i+BATCH_SIZE])
        for j,r in enumerate(res):
            idx=idxs[i+j]
            layers=detector.analyze(r)
            if layers is None:
                continue
            metrics=compute_metrics(layers)
            for k,v in metrics.items():
                df.at[idx,k]=v

    df.to_excel(OUTPUT_FILE,index=False)
    print('Done')
