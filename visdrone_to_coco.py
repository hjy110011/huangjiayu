import os
import json
import cv2
from tqdm import tqdm
from pathlib import Path

# --- 1. 配置信息 ---
# VisDrone11 数据集根目录 (请根据您的实际路径修改)
VISDRONE_ROOT = 'D:\\huangjiayu\\datasets\\VisDrone11'

# COCO JSON 标注文件的输出路径 (通常放在根目录下新建的 annotations 文件夹中)
OUTPUT_DIR = Path(VISDRONE_ROOT) / 'annotations'
OUTPUT_DIR.mkdir(exist_ok=True)

# 定义 VisDrone11 的 10 个有效类别及其 COCO ID (COCO ID从1开始)
# VisDrone11 TXT 中的 category_id (1-based) 对应 COCO JSON 中的 category_id (1-based)
CATEGORY_MAP = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor"
}

# 需要处理的子集及其路径
DATASETS = {
    # 'train': 'VisDrone2019-DET-train',
    # 'val': 'VisDrone2019-DET-val',
    'test-dev': 'VisDrone2019-DET-test-dev' # 通常测试集没有GT，故不转换
}

# VisDrone11 TXT 标注格式字段索引
# <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<category_id>,<truncation>,<occlusion>
IDX_BBOX_LEFT = 2
IDX_BBOX_TOP = 3
IDX_BBOX_WIDTH = 4
IDX_BBOX_HEIGHT = 5
IDX_CATEGORY = 7
IDX_IGNORED = [0, 11]  # VisDrone11 TXT 中类别ID 0 和 11 是忽略/其他，应跳过


# --- 2. 转换函数 ---

def convert_visdrone_to_coco(subset_name, sub_folder):
    """
    将单个 VisDrone11 子集转换为 COCO JSON 格式。
    """
    print(f"--- 🚀 开始转换 {subset_name} 集 ---")

    # 路径设置
    images_dir = Path(VISDRONE_ROOT) / sub_folder / 'images'
    annotations_dir = Path(VISDRONE_ROOT) / sub_folder / 'annotations'
    output_json_path = OUTPUT_DIR / f'instances_{subset_name}2019.json'

    coco_format = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 填充 categories 字段 (基于 CATEGORY_MAP)
    for cat_id, cat_name in CATEGORY_MAP.items():
        coco_format["categories"].append({
            "supercategory": "none",
            "id": cat_id,
            "name": cat_name
        })

    img_id = 1
    ann_id = 1

    # 遍历图像文件夹中的所有图片
    image_files = sorted(os.listdir(images_dir))

    for img_file in tqdm(image_files, desc=f"Converting {subset_name}"):
        if not img_file.endswith(('.jpg', '.png')):
            continue

        # 1. 处理图像信息
        img_path = str(images_dir / img_file)
        # 使用 cv2 读取图像以获取尺寸
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_file}")
            continue
        h, w, _ = img.shape

        image_info = {
            "file_name": f"{sub_folder}/images/{img_file}",  # 相对路径
            "height": h,
            "width": w,
            "id": img_id
        }
        coco_format["images"].append(image_info)

        # 2. 处理标注信息 (VisDrone11 的标注文件名与图片名相同，但扩展名为 .txt)
        anno_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
        anno_path = annotations_dir / anno_file

        if not anno_path.exists():
            # VisDrone11 有些图片可能没有标注文件
            print(f"Warning: Annotation file not found for {img_file}")
            img_id += 1
            continue

        with open(anno_path, 'r') as f:
            for line in f.readlines():
                # VisDrone11 TXT 字段是以逗号分隔的
                parts = line.strip().split(',')
                if len(parts) != 10:
                    continue  # 确保行格式正确

                try:
                    # 解析 VisDrone11 标注
                    bbox_left = int(float(parts[IDX_BBOX_LEFT]))
                    bbox_top = int(float(parts[IDX_BBOX_TOP]))
                    bbox_width = int(float(parts[IDX_BBOX_WIDTH]))
                    bbox_height = int(float(parts[IDX_BBOX_HEIGHT]))
                    visdrone_cat_id = int(parts[IDX_CATEGORY])
                except ValueError:
                    continue  # 跳过解析失败的行

                # 3. 过滤和映射类别
                if visdrone_cat_id in IDX_IGNORED or visdrone_cat_id not in CATEGORY_MAP:
                    continue  # 跳过忽略的类别

                # 4. 转换为 COCO annotation
                # COCO bbox 格式: [x_min, y_min, width, height]
                coco_bbox = [bbox_left, bbox_top, bbox_width, bbox_height]
                area = bbox_width * bbox_height

                # 忽略极小或无效的边界框
                if area <= 0 or bbox_width < 1 or bbox_height < 1:
                    continue

                annotation = {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": visdrone_cat_id,  # 使用映射后的 COCO ID (1-10)
                    "bbox": coco_bbox,
                    "area": area,
                    "iscrowd": 0,  # VisDrone通常不提供is_crowd信息，设为0
                    "segmentation": []  # 目标检测任务，分割字段为空
                }

                coco_format["annotations"].append(annotation)
                ann_id += 1

        img_id += 1

    # 3. 保存 JSON 文件
    print(
        f"\n✅ {subset_name} 转换完成. 总计图片: {len(coco_format['images'])}, 总计标注: {len(coco_format['annotations'])}")
    with open(output_json_path, 'w') as f:
        json.dump(coco_format, f)
    print(f"💾 COCO JSON 文件已保存至: {output_json_path}")


# --- 3. 主函数执行 ---

if __name__ == "__main__":
    for subset, folder in DATASETS.items():
        convert_visdrone_to_coco(subset, folder)