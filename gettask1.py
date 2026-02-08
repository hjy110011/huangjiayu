import os
import random
import xml.etree.ElementTree as ET
from tqdm import tqdm

# ================= 配置区域 =================
VOC_ROOT = "D:/xview/xView_VOC"
ANNOTATIONS_DIR = os.path.join(VOC_ROOT, "Annotations")
OUTPUT_DIR = os.path.join(VOC_ROOT, "ImageSets2", "Main")

# ================= 任务定义 =================

TASK_MAPPING = {
    "t1_train": [
        11, 12, 13, 15, 17, 18, 19, 20, 23, 33, 38, 40, 41, 47
    ],
    "t2_train": [
        21, 24, 25, 26, 27, 28, 29, 32, 34, 35, 36, 37, 49, 50, 51, 52, 45
    ],
    "t3_train": [
        53, 54, 55, 56, 57, 59, 60, 61, 62, 63, 64, 65, 66, 44
    ],
    "t4_train": [
        71, 72, 73, 74, 76, 77, 79, 83, 84, 86, 89, 91, 93, 94, 42
    ]
}

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

# 构建查找表
NAME_TO_TASK = {}
NAME_TO_ID = {}

for task_name, ids in TASK_MAPPING.items():
    for xid in ids:
        raw_name = XVIEW_CLASSES[xid]
        clean_name = raw_name.replace(" ", "_").replace("/", "_")
        NAME_TO_TASK[clean_name] = task_name
        NAME_TO_ID[clean_name] = xid


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    task_files = {t: [] for t in TASK_MAPPING.keys()}
    all_files = []

    # 统计数据： {class_name: image_count}
    # 初始化所有类别计数为 0
    category_stats = {name: 0 for name in NAME_TO_TASK.keys()}

    # 总体筛选统计
    stats = {"mixed_discarded": 0, "empty_discarded": 0, "kept": 0}

    if not os.path.exists(ANNOTATIONS_DIR):
        print(f"错误: 找不到目录 {ANNOTATIONS_DIR}")
        return

    xml_list = [x for x in os.listdir(ANNOTATIONS_DIR) if x.endswith(".xml")]
    print(f"正在扫描 {len(xml_list)} 个 XML 文件 (严格模式 + 类别统计)...")

    # 1. 确定任务顺序
    task_order = ["t1_train", "t2_train", "t3_train", "t4_train"]

    for xml_file in tqdm(xml_list):
        file_id = os.path.splitext(xml_file)[0]
        all_files.append(file_id)

        try:
            tree = ET.parse(os.path.join(ANNOTATIONS_DIR, xml_file))
            root = tree.getroot()

            max_task_idx = -1  # 记录图中类别所属的最高任务索引
            classes_found_in_image = set()

            for obj in root.findall("object"):
                name = obj.find("name").text
                if name in NAME_TO_TASK:
                    classes_found_in_image.add(name)
                    # 获取该类别所属的任务索引
                    current_task_name = NAME_TO_TASK[name]
                    current_idx = task_order.index(current_task_name)

                    # 更新最高索引
                    if current_idx > max_task_idx:
                        max_task_idx = current_idx

            # 2. 分类逻辑
            if max_task_idx != -1:
                # 这张图被归类为它包含的“最高级”任务
                # 例如：图中同时有 T1 和 T2 类别，max_task_idx 为 1 (t2_train)
                assigned_task = task_order[max_task_idx]
                task_files[assigned_task].append(file_id)
                stats["kept"] += 1

                # 统计该图片贡献的类别数
                for cname in classes_found_in_image:
                    category_stats[cname] += 1
            else:
                # 不包含任何我们定义的 60 类目标
                stats["empty_discarded"] += 1

        except Exception as e:
            print(f"警告: 解析文件 {xml_file} 时出错: {e}")
    # ================= 输出 TXT =================
    # 生成 trainval.txt (记录所有扫描过的图片ID)
    with open(os.path.join(OUTPUT_DIR, "trainval.txt"), "w") as f:
        f.write("\n".join(sorted(all_files)))

    sorted_tasks = sorted(task_files.keys())
    for task_name in sorted_tasks:
        ids = task_files[task_name]
        random.shuffle(ids)
        with open(os.path.join(OUTPUT_DIR, f"{task_name}.txt"), "w") as f:
            f.write("\n".join(ids))

    # ================= 输出 统计报表 =================
    print("\n" + "=" * 60)
    print(f"{'TASK':<10} | {'ID':<4} | {'CLASS NAME':<30} | {'IMG COUNT':<10}")
    print("-" * 60)

    for task_name in sorted_tasks:
        # 获取属于该任务的所有 ID，按 ID 排序
        task_ids = sorted(TASK_MAPPING[task_name])

        for xid in task_ids:
            raw_name = XVIEW_CLASSES[xid]
            clean_name = raw_name.replace(" ", "_").replace("/", "_")
            count = category_stats.get(clean_name, 0)

            print(f"{task_name:<10} | {xid:<4} | {clean_name:<30} | {count:<10}")

        print("-" * 60)  # 任务间分隔线

    print("\n" + "=" * 30)
    print(f"筛选汇总:")
    print(f"  保留的纯净图片: {stats['kept']}")
    print(f"  剔除的混合图片: {stats['mixed_discarded']} (包含跨任务目标)")
    print(f"  剔除的无效图片: {stats['empty_discarded']}")
    print(f"  文件已保存在: {OUTPUT_DIR}")
    print("=" * 30 + "\n")


if __name__ == "__main__":
    main()