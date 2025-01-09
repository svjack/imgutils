'''
python score_tag_script.py . three_tag_output
'''

import os
import json
from tqdm import tqdm
from PIL import Image
from ccip import _VALID_MODEL_NAMES, _DEFAULT_MODEL_NAMES, ccip_difference, ccip_default_threshold
from datasets import load_dataset
import pathlib
import argparse
from imgutils.tagging import get_wd14_tags  # 导入 get_wd14_tags 函数

# 加载数据集
Genshin_Impact_Illustration_ds = load_dataset("svjack/Genshin-Impact-Illustration")["train"]
ds_size = len(Genshin_Impact_Illustration_ds)
name_image_dict = {}
for i in range(ds_size):
    row_dict = Genshin_Impact_Illustration_ds[i]
    name_image_dict[row_dict["name"]] = row_dict["image"]

def _compare_with_dataset(imagex, model_name):
    threshold = ccip_default_threshold(model_name)
    results = []

    for name, imagey in name_image_dict.items():
        diff = ccip_difference(imagex, imagey)
        result = {
            "difference": diff,
            "prediction": 'Same' if diff <= threshold else 'Not Same',
            "name": name
        }
        results.append(result)

    # 按照 diff 值进行排序
    results.sort(key=lambda x: x["difference"])

    return results

def process_image(image_path, model_name, output_dir):
    image = Image.open(image_path)
    results = _compare_with_dataset(image, model_name)

    # 获取 WD14 标签
    rating, features, chars = get_wd14_tags(image_path)

    # 构建最终的输出字典
    output_data = {
        "results": results,  # 保存比较结果
        "rating": rating,    # 保存 WD14 的 rating
        "features": features,  # 保存 WD14 的 features
        "characters": chars    # 保存 WD14 的 characters
    }

    # 生成输出文件名
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_file = os.path.join(output_dir, f"{image_name}.json")

    # 保存结果到 JSON 文件
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Compare images with a dataset and save results as JSON.")
    parser.add_argument("input_path", type=str, help="Path to the input image or directory containing images.")
    parser.add_argument("output_dir", type=str, help="Directory to save the output JSON files.")
    parser.add_argument("--model", type=str, default=_DEFAULT_MODEL_NAMES, choices=_VALID_MODEL_NAMES, help="Model to use for comparison.")

    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 判断输入路径是文件还是目录
    if os.path.isfile(args.input_path):
        image_paths = [args.input_path]
    elif os.path.isdir(args.input_path):
        image_paths = list(pathlib.Path(args.input_path).rglob("*.png")) + list(pathlib.Path(args.input_path).rglob("*.jpg"))
    else:
        raise ValueError("Input path must be a valid file or directory.")

    # 处理每个图片
    for image_path in tqdm(image_paths, desc="Processing images"):
        process_image(image_path, args.model, args.output_dir)

if __name__ == '__main__':
    main()