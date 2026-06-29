import pandas as pd
import os
import requests
from tqdm import tqdm

# ===== 路径设置（改成你自己的）=====
excel_path = r"./（6.24)5月冰柜竞赛明细清单-正激励-复核版.xlsx"
old_mapping_path = r"./juice_dataset/data/images/train/mapping.csv"
output_dir = r"./juice_dataset/data/unlabeled_new"

os.makedirs(output_dir, exist_ok=True)

# ===== 读取数据 =====
df = pd.read_excel(excel_path, sheet_name="data")
old_df = pd.read_csv(old_mapping_path)

# ===== 去重（避免重复下载）=====
used_urls = set(old_df['url'].dropna())

df_new = df[~df['照片链接'].isin(used_urls)]

print(f"✅ 剩余未使用图片数量: {len(df_new)}")

# ===== 放宽筛选（优先难样本）=====
df_new['模型误判原因'] = df_new['模型误判原因'].astype(str)

df_filtered = df_new[
    df_new['模型误判原因'].str.contains('果汁', na=False) |
    (df_new['竞赛项目'] == '果汁冰柜')
]

print(f"✅ 筛选后数量: {len(df_filtered)}")

# ===== 抽样 =====
sample_size = 200

if len(df_filtered) > sample_size:
    df_sample = df_filtered.sample(n=sample_size, random_state=42)
else:
    df_sample = df_filtered

print(f"✅ 本次下载数量: {len(df_sample)}")

# ===== 下载函数 =====
def download_image(url, save_path):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(r.content)
            return True
        else:
            print(f"❌ failed: {url}")
            return False
    except Exception as e:
        print(f"❌ error: {url} -> {e}")
        return False


# create mapping data csv
mapping = []
counter = 0

for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    url = row['照片链接']
    if pd.isna(url):
        continue

    filename = f"image{counter}.jpg"
    save_path = os.path.join(output_dir, filename)

    success = download_image(url, save_path)

    if success:
        mapping.append({
            "filename": filename,
            "url": url,
            "竞赛项目": row['竞赛项目'],
            "模型误判原因": row['模型误判原因'],
            "复核问题": row['复核问题']
        })
        counter += 1

# save mapping.csv
mapping_path = os.path.join(output_dir, "mapping.csv")
pd.DataFrame(mapping).to_csv(mapping_path, index=False)
print(f"✅ mapping已保存: {mapping_path}")
print(f"✅ 下载完成，总共 {counter} 张")

