import os,requests
from PIL import Image

def fetch_image(row,idx,cache_dir):
    path=os.path.join(cache_dir,f'{idx}.jpg')
    try:
        if os.path.exists(path):
            return idx,Image.open(path).convert('RGB')
        r=requests.get(row['image_url'],timeout=5)
        with open(path,'wb') as f:
            f.write(r.content)
        return idx,Image.open(path).convert('RGB')
    except:
        return idx,None
