import os
import json
import gc  # 新增：用于垃圾回收
import xml.etree.ElementTree as ET
from tqdm import tqdm

# ================= 配置 =================
VOC_ROOT = "/home/gdut-627/106G/public-dataset/OWOD/xview/xView_VOC"
XML_DIR = os.path.join(VOC_ROOT, "Annotations")
TXT_DIR = os.path.join(VOC_ROOT, "ImageSets", "Main")
OUTPUT_DIR = os.path.join(VOC_ROOT, "SOWOD_Split")  # JSON 输出位置

# ================= 15-15-15-15 划分配置 =================
SOWOD_TASKS = [
    ("t1_train", [11, 12, 13, 15, 17, 18, 19, 20, 23, 33, 34, 38, 40, 41, 47]),
    ("t2_train", [21, 24, 25, 26, 27, 28, 29, 32, 49, 50, 51, 52, 53, 60, 61]),
    ("t3_train", [54, 55, 56, 57, 59, 62, 63, 64, 65, 66, 71, 72, 79, 83, 84]),
    ("t4_train", [73, 74, 76, 77, 86, 89, 91, 93, 94, 36, 37, 35, 42, 44, 45])
]

# xView 原始ID -> 名称映射
XVIEW_CLASSES = {
    11: 'Fixed-wing Aircraft', 12: 'Small Aircraft', 13: 'Cargo Plane', 15: 'Helicopter',
    17: 'Passenger Vehicle', 18: 'Small Car', 19: 'Bus', 20: 'Pickup Truck',
    21: 'Utility Truck', 23: 'Truck', 24: 'Cargo Truck', 25: 'Truck w/Box',
    26: 'Truck Tractor', 27: 'Trailer', 28: 'Truck w/Flatbed', 29: 'Truck w/Liquid',
    32: 'Crane Truck', 33: 'Railway Vehicle', 34: 'Passenger Car', 35: 'Cargo Car',
    36: 'Flat Car', 37: 'Tank car', 38: 'Locomotive', 40: 'Maritime Vessel',
    41: 'Motorboat', 42: 'Sailboat', 44: 'Tugboat', 45: 'Barge', 47: 'Fishing Vessel',
    49: 'Ferry', 50: 'Yacht', 51: 'Container Ship', 52: 'Oil Tanker',
    53: 'Engineering Vehicle', 54: 'Tower crane', 55: 'Container Crane',
    56: 'Reach Stacker', 57: 'Straddle Carrier', 59: 'Mobile Crane', 60: 'Dump Truck',
    61: 'Haul Truck', 62: 'Scraper/Tractor', 63: 'Front loader/Bulldozer',
    64: 'Excavator', 65: 'Cement Mixer', 66: 'Ground Grader', 71: 'Hut/Tent',
    72: 'Shed', 73: 'Building', 74: 'Aircraft Hangar', 76: 'Damaged Building',
    77: 'Facility', 79: 'Construction Site', 83: 'Vehicle Lot', 84: 'Helipad',
    86: 'Storage Tank', 89: 'Shipping container lot', 91: 'Shipping Container',
    93: 'Pylon', 94: 'Tower'
}


def build_global_mapping():
    """构建全局 ID 映射: Name -> 0~59"""
    name_to_id = {}
    id_to_name = {}
    new_id_counter = 0
    full_class_names = []

    for task_name, xids in SOWOD_TASKS:
        for xid in xids:
            name = XVIEW_CLASSES[xid].replace(" ", "_").replace("/", "_")
            name_to_id[name] = new_id_counter
            id_to_name[new_id_counter] = name
            full_class_names.append(name)
            new_id_counter += 1

    return name_to_id, id_to_name, full_class_names


NAME_TO_ID, ID_TO_NAME, FULL_CLASS_NAMES = build_global_mapping()


def convert_to_coco(subset_name, txt_file, output_json, allowed_ids=None):
    if not os.path.exists(txt_file):
        print(f"Skipping {subset_name}, {txt_file} not found.")
        return

    with open(txt_file, 'r') as f:
        img_ids = [x.strip() for x in f.readlines() if x.strip()]

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": cid, "name": ID_TO_NAME[cid], "supercategory": "none"} for cid in range(len(ID_TO_NAME))]
    }

    ann_id = 1

    for idx, file_id in enumerate(tqdm(img_ids, desc=f"Generating {subset_name}")):
        xml_path = os.path.join(XML_DIR, f"{file_id}.xml")
        if not os.path.exists(xml_path): continue

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Image info
            w = int(root.find("size/width").text)
            h = int(root.find("size/height").text)
            file_name = root.find("filename").text

            coco_data["images"].append({
                "id": idx + 1,
                "file_name": file_name,
                "height": h,
                "width": w
            })

            # Annotation info
            for obj in root.findall("object"):
                cname = obj.find("name").text

                if cname not in NAME_TO_ID: continue
                cid = NAME_TO_ID[cname]

                # [核心过滤逻辑]
                if allowed_ids is not None and cid not in allowed_ids:
                    continue

                bnd = obj.find("bndbox")
                xmin, ymin = int(bnd.find("xmin").text), int(bnd.find("ymin").text)
                xmax, ymax = int(bnd.find("xmax").text), int(bnd.find("ymax").text)
                bw, bh = xmax - xmin, ymax - ymin

                if bw <= 0 or bh <= 0: continue

                coco_data["annotations"].append({
                    "id": ann_id,
                    "image_id": idx + 1,
                    "category_id": cid,
                    "bbox": [xmin, ymin, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                    "ignore": 0,
                    "segmentation": []
                })
                ann_id += 1

            # 【重要】显式释放 XML 树，防止内存堆积
            del root, tree
            if idx % 1000 == 0:
                gc.collect()

        except Exception as e:
            print(f"Error parsing {file_id}: {e}")

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(coco_data, f)

    # 彻底释放大字典内存
    num_imgs = len(coco_data['images'])
    num_anns = len(coco_data['annotations'])
    del coco_data
    gc.collect()

    print(f"Saved: {output_json} (Images: {num_imgs}, Anns: {num_anns})")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    global_offset = 0
    cumulative_ids = set()  # 【修改点 1】创建一个累积集合

    for i, (task_name, xids) in enumerate(SOWOD_TASKS):
        task_num = i + 1
        txt_path = os.path.join(TXT_DIR, f"{task_name}.txt")
        json_path = os.path.join(OUTPUT_DIR, f"t{task_num}_train.json")

        num_classes = len(xids)

        # 【修改点 2】将当前任务的类别 ID 加入到累积集合中
        current_task_ids = range(global_offset, global_offset + num_classes)
        cumulative_ids.update(current_task_ids)

        print(f"\n--- Processing Task {task_num} (Cumulative Classes: {len(cumulative_ids)}) ---")

        # 【修改点 3】传入的是累积的 cumulative_ids，而不是 current_task_ids
        convert_to_coco(f"t{task_num}", txt_path, json_path, allowed_ids=cumulative_ids)

        global_offset += num_classes

    # 生成 Test 数据 (包含所有图片，所有类别)
    print("\n--- Processing Test Set (All Classes) ---")
    test_txt = os.path.join(TXT_DIR, "trainval.txt")
    test_json = os.path.join(OUTPUT_DIR, "test.json")
    convert_to_coco("test", test_txt, test_json, allowed_ids=None)


if __name__ == "__main__":
    main()