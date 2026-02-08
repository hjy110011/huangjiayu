import json
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 数据集根目录 (请根据你的实际路径修改)
# 假设你的目录结构是 data/VOC2007/ImageSets/Main/t1_train.txt
DATA_ROOT = "D:/UAV-OWD/SOWOD_Merged_VOC"
ANNOTATIONS_DIR = os.path.join(DATA_ROOT, "Annotations")
IMAGESET_DIR = os.path.join(DATA_ROOT, "ImageSets", "Tasks")  # txt 文件通常在这里
OUTPUT_DIR = os.path.join(DATA_ROOT,"COCO_JSONB")  # json 输出到这里

# 2. 定义需要转换的任务映射
# 格式: {"输入txt文件名": "输出json文件名"}
FILES_TO_CONVERT = {
    # 训练集
    "ScenarioB_Task1.txt": "t1_train.json",
    "ScenarioB_Task2.txt": "t2_train.json",
    "ScenarioB_Task3.txt": "t3_train.json",
    "ScenarioB_Task4.txt": "t4_train.json",
    "all_cleaned.txt":"test.json"
}

# 3. 任务类别定义 (与 config 必须严格一致)
T1_CLASSES = [
    "airplane", "ship", "car", "truck", "bus", "van", "bridge",
    "harbor", "storage-tank", "chimney", "pedestrian", "people",
    "bicycle", "motor"
]
T2_CLASSES = [
    "baseball-field", "basketball-court", "tennis-court",
    "ground-track-field", "soccer-ball-field", "swimming-pool", "stadium"
]
T3_CLASSES = [
    "airport", "train-station", "dam", "overpass",
            "toll-station", "service-area", "roundabout"

]
T4_CLASSES = [
    "helicopter", "windmill", "golf-field",
    "tricycle", "awning-tricycle"
]

# 4. 构建全局类别映射
# 注意：生成的 json 会包含所有 80 个类别定义，但在 annotation 中只保留该任务相关的
ALL_CLASSES = T1_CLASSES + T2_CLASSES + T3_CLASSES + T4_CLASSES
CAT_NAME_TO_ID = {name: i for i, name in enumerate(ALL_CLASSES)}


# 辅助函数：确定当前 txt 文件属于哪个任务，从而决定 target_classes
def get_target_classes(filename):
    # 如果是 Task 1，只保留 T1 类别
    if "t1_" in filename:
        return set(T1_CLASSES)
    # 如果是 Task 2 (包括 t2_train 和 t2_ft)，通常需要检测 T1+T2 类别
    # 因为 t2_ft 可能包含旧类回放。
    # 为了保险，对于 Task 2及以后，我们允许检测所有 已知 类别
    elif "t2_" in filename:
        return set(T1_CLASSES + T2_CLASSES)
    elif "t3_" in filename:
        return set(T1_CLASSES + T2_CLASSES + T3_CLASSES)
    elif "t4_" in filename:
        return set(ALL_CLASSES)
    elif "our" in filename:
        return set(ALL_CLASSES)  # 测试集包含所有
    else:
        return set(ALL_CLASSES)  # 默认所有


# ================= 核心逻辑 =================

def convert_txt_to_json(txt_filename, json_filename):
    txt_path = os.path.join(IMAGESET_DIR, txt_filename)
    json_path = os.path.join(OUTPUT_DIR, json_filename)

    if not os.path.exists(txt_path):
        print(f"[Skip] {txt_filename} not found.")
        return

    print(f"Converting {txt_filename} -> {json_filename} ...")

    # 确定目标类别集合
    target_classes = get_target_classes(txt_filename)

    with open(txt_path, 'r') as f:
        lines = f.readlines()

    # 解析图片ID (处理可能的后缀)
    image_ids = []
    for line in lines:
        content = line.strip().split(' ')[0]  # 只要第一列
        if content.lower().endswith('.jpg'):
            content = content[:-4]  # 去掉 .jpg，因为后面我们要用来找 xml
        image_ids.append(content)

    coco_output = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 生成 Categories
    for name in ALL_CLASSES:
        coco_output["categories"].append({
            "id": CAT_NAME_TO_ID[name],
            "name": name
        })

    ann_id = 0

    for img_id_str in tqdm(image_ids):
        xml_path = os.path.join(ANNOTATIONS_DIR, f"{img_id_str}.xml")

        if not os.path.exists(xml_path):
            # print(f"Warning: XML not found for {img_id_str}") # 避免刷屏
            continue

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except:
            continue

        # 图片信息
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        # 尝试将文件名转为数字ID，如果失败则hash
        try:
            # 假设文件名是 2008_00001 这种格式
            # 移除下划线尝试转int，或者直接取后几位
            # 为了唯一性，简单处理：
            if "_" in img_id_str:
                image_int_id = int(img_id_str.split('_')[-1])
            else:
                image_int_id = int(img_id_str)
        except:
            image_int_id = int(hash(img_id_str) % 10000000)

        coco_output["images"].append({
            "file_name": f"{img_id_str}.jpg",  # 强制加上 .jpg
            "height": height,
            "width": width,
            "id": image_int_id
        })

        # 标注信息
        for obj in root.findall("object"):
            cls_name = obj.find("name").text

            # 过滤：只保留当前任务范围内的类别
            if cls_name not in target_classes:
                continue

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            w = xmax - xmin
            h = ymax - ymin

            coco_output["annotations"].append({
                "id": ann_id,
                "image_id": image_int_id,
                "category_id": CAT_NAME_TO_ID[cls_name],
                "bbox": [xmin, ymin, w, h],
                "area": w * h,
                "iscrowd": 0,
                "segmentation": []
            })
            ann_id += 1

    # 保存
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    with open(json_path, 'w') as f:
        json.dump(coco_output, f)
    print(f"Saved {json_filename}. Images: {len(coco_output['images'])}, Anns: {len(coco_output['annotations'])}")


if __name__ == "__main__":
    for txt, json_name in FILES_TO_CONVERT.items():
        convert_txt_to_json(txt, json_name)