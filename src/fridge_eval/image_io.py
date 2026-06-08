import os
import requests
from PIL import Image
import time

def fetch_image(row, idx, cache_dir):
    path = os.path.join(cache_dir, f"{idx}.jpg")

    try:
        if os.path.exists(path):
            return idx, Image.open(path).convert("RGB")

        session = requests.Session()
        session.trust_env = False

        for _ in range(2):
            try:
                r = session.get(
                    row["image_url"],
                    timeout=(2, 3),
                    stream=True
                )

                if r.status_code != 200:
                    continue

                content = r.raw.read(3 * 1024 * 1024, decode_content=True)

                with open(path, "wb") as f:
                    f.write(content)

                img = Image.open(path).convert("RGB")

                time.sleep(0.01)

                return idx, img

            except:
                continue

        return idx, None

    except:
        return idx, None
