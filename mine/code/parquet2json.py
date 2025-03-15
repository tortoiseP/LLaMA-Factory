import pandas as pd
import json
import os

# 输入文件路径
input_file = '/workspace/datasets/llava-en-zh-300k/zh/train-00000-of-00059.parquet'

# 输出文件夹路径
output_folder = '/workspace/datasets/llava-en-zh-300k/transformed'
images_folder = os.path.join(output_folder, 'images')

# 确保输出文件夹存在
os.makedirs(images_folder, exist_ok=True)

# 读取 parquet 文件
df = pd.read_parquet(input_file)

# 函数：将字节数据保存为图像文件
def save_image(image_bytes, image_name):
    image_path = os.path.join(images_folder, image_name)
    with open(image_path, 'wb') as img_file:
        img_file.write(image_bytes)
    return image_path

# 构造 JSON 格式的数据
transformed_data = []

for idx, row in df.iterrows():
    record = {
        "messages": [],
        "images": []
    }
    # 处理 messages
    messages = row['messages']
    for msg in messages:
        record['messages'].append({
            "content": msg['content'],
            "role": msg['role']
        })
    # 处理 images（将字节数据保存为文件）
    for i, image_bytes in enumerate(row['images']):
        image_name = f"{idx + 1}_{i+1}.jpg"  # 生成唯一的图像名称
        image_path = save_image(image_bytes['bytes'], image_name)
        record['images'].append(f"images/{image_name}")
    transformed_data.append(record)

# 输出文件路径
output_file = os.path.join(output_folder, 'llava-zh-3k.json')
# 保存为 JSON 文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(transformed_data, f, ensure_ascii=False, indent=2)
print(f"转换完成，文件保存在: {output_file}")
