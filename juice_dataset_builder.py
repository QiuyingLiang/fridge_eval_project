import pandas as pd
import os
import requests
from tqdm import tqdm

# path
excel_path = r"./（6.24)5月冰柜竞赛明细清单-正激励-复核版.xlsx"
output_dir = r"./juice_dataset/data/images/train"

os.makedirs(output_dir, exist_ok=True)

# read data
df = pd.read_excel(excel_path, sheet_name="data")


df['模型误判原因'] = df['模型误判原因'].astype(str)
df['复核问题'] = df['复核问题'].astype(str)

juice_df = df[df['竞赛项目'] == '果汁冰柜']
juice_df = juice_df.sample(n=min(60, len(juice_df)), random_state=42)

mis_df = df[
    (df['竞赛项目'] == '非果汁冰柜') &
    (df['模型误判原因'].str.contains('果汁', na=False))
]
mis_df = mis_df.sample(n=min(50, len(mis_df)), random_state=42)

noise_df = df[
    df['复核问题'].str.contains('花心|空心', na=False)
]
noise_df = noise_df.sample(n=min(30, len(noise_df)), random_state=42)

hard_df = df[
    (df['竞赛项目'] == '非果汁冰柜') &
    (df['模型误判原因'] == 'nan')
]
hard_df = hard_df.sample(n=min(20, len(hard_df)), random_state=42)

# combine data
final_df = pd.concat([juice_df, mis_df, noise_df, hard_df])
final_df = final_df.drop_duplicates(subset=['照片链接'])

print(f"✅ 最终样本数: {len(final_df)}")

# download
def download_image(url, save_path):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(r.content)
            return True
        else:
            print(f"❌ 失败: {url} 状态码={r.status_code}")
            return False
    except Exception as e:
        print(f"❌ 异常: {url} -> {e}")
        return False

# create mapping data csv
mapping = []
counter = 0

for _, row in tqdm(final_df.iterrows(), total=len(final_df)):
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

print(f"✅ 下载完成，共 {counter} 张")
print(f"✅ mapping已保存: {mapping_path}")
