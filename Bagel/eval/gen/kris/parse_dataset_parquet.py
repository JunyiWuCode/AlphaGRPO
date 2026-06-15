from tqdm import tqdm
import pandas as pd
import os
import json
import io
from PIL import Image

# 1. 设置路径
parquet_path = './data/.parquet'
root = './data/'
json_path = "./.json"
print("Loading Parquet file...")
df = pd.read_parquet(parquet_path)
data_map = {row['category'] + '_' + str(row['id']): row for row in df.to_dict('records')}
print(f"Parquet loaded. Found {len(data_map)} records.")

# 3. 读取 Metadata JSON
print("Loading Metadata JSON...")
with open(json_path, 'r', encoding='utf-8') as f:
    metadata_list = json.load(f)

def save_image_from_bytes(image_data, save_path):
    """保存图片的辅助函数"""
    try:
        b_data = None
        # 处理 {'bytes': b'...'} 格式
        if isinstance(image_data, dict) and 'bytes' in image_data:
            b_data = image_data['bytes']
        # 处理直接是 bytes 的格式
        elif isinstance(image_data, bytes):
            b_data = image_data
            
        if b_data:
            img = Image.open(io.BytesIO(b_data))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(save_path)
            return True
    except Exception as e:
        print(f"Error saving {save_path}: {e}")
    return False

# 4. 遍历 Metadata 并保存图片
# print("Start extracting images based on metadata...")

# for item in tqdm(metadata_list, desc="Processing metadata items"):
#     pid = item["type"] + '_' + str(item['id'])

#     category = item['type'] # 对应 parquet 里的 category，用于创建文件夹
    
#     # 获取 Parquet 中对应的数据行
#     row_data = data_map.get(pid)
#     if not row_data:
#         print(f"Warning: ID {pid} found in metadata but not in parquet file. Skipping.")
#         continue

#     # 确保存储目录存在
#     save_dir = os.path.join(root, category)
#     os.makedirs(save_dir, exist_ok=True)

#     # --- 处理 ori_img (输入图片) ---
#     ori_img_info = item['ori_img']
    
#     if isinstance(ori_img_info, str):
#         # 情况 A: ori_img 是字符串 -> 说明只有 image_1
#         # 文件名：metadata 中指定的字符串
#         save_path = os.path.join(save_dir, ori_img_info)
#         # 数据源：row_data['image_1']
#         save_image_from_bytes(row_data.get('image'), save_path)
        
#     elif isinstance(ori_img_info, list):
#         # 情况 B: ori_img 是列表 -> 说明有 image_1, image_2...
#         # 遍历列表，文件名取列表中的名字，数据源依次取 image_1, image_2...
#         for idx, filename in enumerate(ori_img_info):
#             if idx == 0:
#                 col_name = 'image'
#             else:
#                 col_name = f'image_{idx}' # 对应 image_1, image_2, etc.
#             save_path = os.path.join(save_dir, filename)
#             save_image_from_bytes(row_data.get(col_name), save_path)

#     # --- 处理 gt_img (Ground Truth) ---
#     # 只有当 metadata 里明确写了 gt_img 时才保存
#     if 'gt_img' in item and item['gt_img']:
#         gt_filename = item['gt_img']
#         save_path = os.path.join(save_dir, gt_filename)
#         # 数据源：row_data['gt_image']
#         save_image_from_bytes(row_data.get('gt_image'), save_path)

from collections import defaultdict
category2meta = defaultdict(dict)
for item in tqdm(metadata_list, desc="Processing metadata items"):
    category = item['type']
    category2meta[category][item['id']] = item

for category in category2meta:
    json.dump(category2meta[category], open(os.path.join(root, category, f'annotation.json'), 'w'), ensure_ascii=False, indent=2)

print("All images processed.")