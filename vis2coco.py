import os
import json
import gc
import xml.etree.ElementTree as ET
from tqdm import tqdm

# ================= 配置区域 =================
# 数据集根目录
ROOT_DIR = r"D:/UAV-OWD/SOWOD_Merged_VOC"

# 输入路径
XML_DIR = os.path.join(ROOT_DIR, "Annotations")
TXT_DIR = os.path.join(ROOT_DIR, "ImageSets", "Tasks")  # 读取任务划分的txt
GLOBAL_TXT_DIR = os.path.join(ROOT_DIR, "ImageSets", "Main")  # 读取全局test.txt

# 输出路径
OUTPUT_DIR = os.path.join(ROOT_DIR, "COCO_JSONB")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= SOWOD 任务与类别定义 =================
# 必须严格保持顺序，ID将从0开始依次递增
SOWOD_TASKS = [
    # Task 1: Base (14 classes)
    {
        "name": "t1",
        "classes": [
            'airplane', 'ship', 'car', 'truck', 'bus', 'van', 'bridge',
            'harbor', 'storage-tank', 'chimney', 'pedestrian', 'people',
            'bicycle', 'motor'
        ]
    },
    # Task 2: Sports (7 classes)
    {
        "name": "t2",
        "classes": [
            'baseball-field', 'basketball-court', 'tennis-court',
            'ground-track-field', 'soccer-ball-field', 'swimming-pool', 'stadium'
        ]
    },
    # Task 3: Infra (7 classes)
    {
        "name": "t3",
        "classes": [
            'airport', 'train-station', 'dam', 'overpass',
            'toll-station', 'service-area', 'roundabout'
        ]
    },
    # Task 4: Rare (5 classes)
    {
        "name": "t4",
        "classes": [
            'helicopter', 'windmill', 'golf-field',
            'tricycle', 'awning-tricycle'
        ]
    }
]

# 构建全局映射: Name -> ID (0 to 32)
NAME_TO_ID = {}
ID_TO_NAME = {}
FULL_CLASS_NAMES = []


def build_mapping():
    current_id = 0
    for task in SOWOD_TASKS:
        for cls_name in task["classes"]:
            NAME_TO_ID[cls_name] = current_id
            ID_TO_NAME[current_id] = cls_name
            FULL_CLASS_NAMES.append(cls_name)
            current_id += 1
    print(f"Mapping Built. Total Classes: {len(NAME_TO_ID)}")
    print(f"Task 1 Range: 0 - {len(SOWOD_TASKS[0]['classes']) - 1}")


build_mapping()


# ================= 核心转换函数 =================

def convert_to_coco(subset_name, txt_file, output_json, allowed_ids=None):
    """
    将指定的 txt 文件列表中的图片转换为 COCO JSON。
    subset_name: 任务名称 (用于显示进度)
    txt_file: 包含图片ID的txt路径
    output_json: 输出json路径
    allowed_ids: (Set) 允许保留的类别ID。如果为None，则保留所有。
                 用于 OWD 训练，过滤掉属于"未来任务"的标注。
    """
    if not os.path.exists(txt_file):
        print(f"Error: {txt_file} not found. Skipping {subset_name}.")
        return

    # 读取图片 ID 列表
    with open(txt_file, 'r') as f:
        img_ids = [x.strip() for x in f.readlines() if x.strip()]

    # 初始化 COCO 结构
    # 注意：categories 表通常包含所有类，还是只包含当前已知类？
    # 标准做法是 categories 包含所有可能得 ID 定义，但在 annotations 里只出现 allowed_ids
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": cid, "name": ID_TO_NAME[cid], "supercategory": "object"}
            for cid in range(len(ID_TO_NAME))
        ]
    }

    ann_id = 1  # 全局唯一标注 ID

    for idx, file_id in enumerate(tqdm(img_ids, desc=f"Converting {subset_name}")):
        xml_path = os.path.join(XML_DIR, f"{file_id}.xml")
        if not os.path.exists(xml_path):
            continue

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 1. 解析图像信息
            size_node = root.find("size")
            w = int(size_node.find("width").text)
            h = int(size_node.find("height").text)

            # 确保文件名后缀正确
            file_name = root.find("filename").text
            if not file_name.endswith('.jpg'):
                file_name += '.jpg'

            coco_data["images"].append({
                "id": idx + 1,  # Image ID 从 1 开始
                "file_name": file_name,
                "height": h,
                "width": w
            })

            # 2. 解析标注信息
            for obj in root.findall("object"):
                cname = obj.find("name").text

                # 映射类别
                if cname not in NAME_TO_ID:
                    continue  # 忽略不在列表中的杂类

                cid = NAME_TO_ID[cname]

                # [SOWOD 核心]: 过滤掉不在当前已知任务列表中的类别
                # 例如：在 Task1 训练时，如果图片里有 "Helicopter" (Task4)，它将被忽略（视为背景/未知）
                if allowed_ids is not None and cid not in allowed_ids:
                    continue

                bnd = obj.find("bndbox")
                xmin = float(bnd.find("xmin").text)
                ymin = float(bnd.find("ymin").text)
                xmax = float(bnd.find("xmax").text)
                ymax = float(bnd.find("ymax").text)

                width = xmax - xmin
                height = ymax - ymin

                if width <= 0 or height <= 0:
                    continue

                coco_data["annotations"].append({
                    "id": ann_id,
                    "image_id": idx + 1,
                    "category_id": cid,
                    "bbox": [xmin, ymin, width, height],  # COCO 格式: [x, y, w, h]
                    "area": width * height,
                    "iscrowd": 0,
                    "ignore": 0,
                    "segmentation": []  # 目标检测通常为空
                })
                ann_id += 1

            # 内存优化
            del root, tree

        except Exception as e:
            print(f"Error processing {file_id}: {e}")

    # 保存 JSON
    with open(output_json, 'w') as f:
        json.dump(coco_data, f)

    print(f"Saved {output_json}: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations.")

    # 垃圾回收
    del coco_data
    gc.collect()


# ================= 主程序 =================

def main():
    cumulative_ids = set()  # 累积已知类别集合

    # 1. 循环生成各任务的训练集 JSON
    for i, task_info in enumerate(SOWOD_TASKS):
        task_idx = i + 1
        task_name = task_info["name"]

        # 获取当前任务的 Class IDs
        current_task_ids = [NAME_TO_ID[n] for n in task_info["classes"]]

        # SOWOD 训练逻辑：当前已知类 = 之前所有类 + 当前新类
        cumulative_ids.update(current_task_ids)

        print(f"\n>>> Processing Task {task_idx} ({task_name})")
        print(f"Allowed Classes (Count: {len(cumulative_ids)}): {sorted(list(cumulative_ids))}")

        # 读取 TXT: 使用 ScenarioA (增量兼容) 的训练数据
        # 路径: ImageSets/Tasks/ScenarioA_Task1_train.txt
        txt_name = f"ScenarioB_Task{task_idx}.txt"
        txt_path = os.path.join(TXT_DIR, txt_name)

        # 输出: t1_train.json
        json_path = os.path.join(OUTPUT_DIR, f"instances_train_t{task_idx}.json")

        # 执行转换
        convert_to_coco(f"Train_T{task_idx}", txt_path, json_path, allowed_ids=cumulative_ids)

    # 2. 生成全量测试集 JSON (Test)
    # 通常 OWD 测试是在全量测试集上进行的，或者按任务划分的测试集
    # 这里我们生成一个包含所有类别的 test.json，对应全局 test.txt
    print("\n>>> Processing Global Test Set")
    test_txt_path = os.path.join(GLOBAL_TXT_DIR, "test.txt")
    test_json_path = os.path.join(OUTPUT_DIR, "instances_test.json")

    # 测试集不需要过滤 ID (allowed_ids=None)，我们需要评估所有类
    convert_to_coco("Global_Test", test_txt_path, test_json_path, allowed_ids=None)


if __name__ == "__main__":
    main()