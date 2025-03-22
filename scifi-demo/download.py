import os
import sys
import requests
import glob

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(save_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)


# 下载finetune数据集
url = "https://huggingface.co/datasets/zxbsmk/webnovel_cn/resolve/main/novel_cn_token512_50k.json?download=true"
save_path = "data_finetune/scifi-finetune.json"
if not os.path.exists(save_path):
    print("Downloading file...")
    download_file(url, save_path)
    print("File downloaded successfully.")
else:
    print("File already exists.")


# 下载训练数据集合：小说合集
url = "https://huggingface.co/datasets/wzy816/scifi/resolve/main/data.zip?download=true"
save_path = "data/data.zip"
if not os.path.exists(save_path):
    print("Downloading file...")
    download_file(url, save_path)
    print("File downloaded successfully.")
else:
    print("File already exists.")


# 解压训练数据集的zip文件后，将所有txt文件合并成一个文件
def find_txt_files(folder_path):
    return glob.glob(os.path.join(folder_path, '**', '*.txt'), recursive=True)

def concatenate_txt_files(txt_files, output_file):
    with open(output_file, 'w', encoding='utf-8') as output:
        for txt_file in txt_files:
            with open(txt_file, 'r', encoding='utf-8') as file:
                output.write(file.read())
                output.write('\n')

directiory = 'data'
output_file = 'data/scifi.all'
txt_files = find_txt_files(directiory)
print(f"总文件数量:{len(txt_files)}")
print(txt_files[:5])
concatenate_txt_files(txt_files, output_file)
