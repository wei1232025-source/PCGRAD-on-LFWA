# download_data.py
import os
import urllib.request

try:
    import datasets
except ImportError:
    import subprocess
    import sys
    print("Installing Hugging Face 'datasets' library...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
    import datasets

def download_and_save():
    # 1. 从 Hugging Face 下载 LFW 原始数据
    print("Downloading 'bitmind/lfw' from Hugging Face...")
    dataset = datasets.load_dataset("bitmind/lfw", split="train")

    # 将数据集序列化保存到本地，避免后续重复请求网络
    save_dir = "lfw_dataset_local"
    print(f"Saving dataset locally to: {save_dir}")
    dataset.save_to_disk(save_dir)
    print("Dataset saved successfully.")

    # 2. 从官方源或备份源下载属性标注文件
    attr_path = "lfw_attributes.txt"
    url_primary = "http://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt"
    backup_urls = [
        "https://raw.githubusercontent.com/harry771/deep-learning-project/master/lfw_attributes.txt",
        "https://raw.githubusercontent.com/swghosh/Deep-Learning/master/lfw_attributes.txt"
    ]

    if os.path.exists(attr_path):
        print(f"'{attr_path}' already exists.")
        return

    print(f"Downloading attributes file from primary URL: {url_primary}...")
    try:
        req = urllib.request.Request(url_primary, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as r, open(attr_path, "wb") as f:
            f.write(r.read())
        print("Successfully downloaded attributes file.")
    except Exception as e:
        print(f"Primary URL failed: {e}. Trying backups...")
        for b_url in backup_urls:
            try:
                print(f"Trying backup: {b_url}")
                req = urllib.request.Request(b_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as r, open(attr_path, "wb") as f:
                    f.write(r.read())
                print("Successfully downloaded attributes file from backup URL.")
                break
            except Exception as e2:
                print(f"Backup failed: {e2}")

if __name__ == "__main__":
    download_and_save()