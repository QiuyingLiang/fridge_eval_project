import pandas as pd
import os
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def prepare_labeling_dataset():
    """准备需要标注的图片"""
    
    # 1. 读取数据
    df = pd.read_excel('data/input.xlsx')
    print(f"总图片数: {len(df)}")
    
    # 2. 采样 500 张图片
    sample_df = df.sample(n=min(500, len(df)))
    print(f"采样了 {len(sample_df)} 张图片用于标注")
    
    # 3. 保存 URL 列表
    sample_df[['image_url']].to_csv('data/to_label.csv', index=False)
    
    # 4. 下载图片到本地
    download_images(sample_df['image_url'].tolist())
    
    print("\n✅ 准备完成！")
    print("下一步：")
    print("1. 图片已下载到 data/images/ 文件夹")
    print("2. 安装 LabelImg: pip install labelImg")
    print("3. 运行 labelImg data/images/ data/labels/")
    print("4. 标注所有脉动瓶子，标签命名为 'maidong'")

def download_images(urls, max_workers=10):
    """下载图片到本地"""
    os.makedirs('data/images', exist_ok=True)
    os.makedirs('data/labels', exist_ok=True)
    
    def download_one(url_info):
        idx, url = url_info
        try:
            r = requests.get(url, timeout=10)
            with open(f'data/images/img_{idx:04d}.jpg', 'wb') as f:
                f.write(r.content)
            return True
        except Exception as e:
            print(f"下载失败 {idx}: {e}")
            return False
    
    # 使用 enumerate 获取索引
    url_list = list(enumerate(urls))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(download_one, url_list), total=len(url_list), desc="下载图片"))
    
    print(f"下载完成: {sum(results)}/{len(urls)} 成功")

if __name__ == "__main__":
    prepare_labeling_dataset()