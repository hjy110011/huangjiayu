import os
import random
import xml.etree.ElementTree as ET
from tqdm import tqdm

# ================= 配置区域 =================
VOC_ROOT = "D:\\xview\\xView_VOC"
ANNOTATIONS_DIR = os.path.join(VOC_ROOT, "Annotations")
OUTPUT_DIR = os.path.join(VOC_ROOT, "ImageSets", "Main")

# ================= 15-15-15-15 均衡划分定义 =================
# 这里的 ID 是 xView 原始的 type_id
# 逻辑：Task 1 (通用), Task 2 (物流), Task 3 (工程), Task 4 (静态/补充)

TASK_MAPPING = {
    "t1_train": [
        11, 12, 13, 15, 17, 18, 19, 20, 23, 33, 34, 38, 40, 41, 47
    ],
    "t2_train": [
        21, 24, 25, 26, 27, 28, 29, 32, 49, 50, 51, 52, 53, 60, 61
    ],
    "t3_train": [
        54, 55, 56, 57, 59, 62, 63, 64, 65, 66, 71, 72, 79, 83, 84
    ],
    "t4_train": [
        73, 74, 76, 77, 86, 89, 91, 93, 94, 36, 37, 35, 42, 44, 45
        # 注: ID 35 (Cargo Car) 已移至此处以凑齐 15 类
    ]
}

# 原始 ID 到 名称 的映射 (用于解析 XML 中的 name 标签)
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

# 构建 "清洗后的类别名" -> "任务名" 的快速查找表
# 因为 step 1 生成的 xml 里 name 是类似 "Small_Car" 这样的字符串
NAME_TO_TASK = {}
for task_name, ids in TASK_MAPPING.items():
    for xid in ids:
        raw_name = XVIEW_CLASSES[xid]
        # 必须与 step 1 的命名逻辑完全一致：空格转下划线，斜杠转下划线
        clean_name = raw_name.replace(" ", "_").replace("/", "_")
        NAME_TO_TASK[clean_name] = task_name


def main():
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 初始化存储容器
    task_files = {t: [] for t in TASK_MAPPING.keys()}
    all_files = []  # 用于 trainval.txt

    # 获取所有 XML 文件
    if not os.path.exists(ANNOTATIONS_DIR):
        print(f"错误: 找不到标注目录 {ANNOTATIONS_DIR}")
        return

    xml_list = [x for x in os.listdir(ANNOTATIONS_DIR) if x.endswith(".xml")]
    print(f"正在扫描 {len(xml_list)} 个 XML 文件...")

    for xml_file in tqdm(xml_list):
        file_id = os.path.splitext(xml_file)[0]  # 去除 .xml 后缀
        all_files.append(file_id)

        try:
            tree = ET.parse(os.path.join(ANNOTATIONS_DIR, xml_file))
            root = tree.getroot()

            # 记录这张图片里出现过的所有任务类型
            tasks_in_this_image = set()

            for obj in root.findall("object"):
                name = obj.find("name").text
                # 查找这个物体属于哪个任务
                if name in NAME_TO_TASK:
                    tasks_in_this_image.add(NAME_TO_TASK[name])

            # 将图片分配给对应的任务列表
            # 注意：如果一张图既有 Task1 的车，又有 Task3 的挖掘机
            # 它会同时出现在 Task1.txt 和 Task3.txt 中 (这是正确的)
            for task in tasks_in_this_image:
                task_files[task].append(file_id)

        except Exception as e:
            print(f"警告: 解析文件 {xml_file} 时出错: {e}")

    # ================= 写入 TXT 文件 =================
    print("\n正在生成 TXT 文件...")

    # 1. 生成 trainval.txt (包含所有图片)
    trainval_path = os.path.join(OUTPUT_DIR, "trainval.txt")
    with open(trainval_path, "w") as f:
        f.write("\n".join(sorted(all_files)))
    print(f"  [Total] trainval.txt: {len(all_files)} 张图片")

    # 2. 生成各任务的 txt
    # 强制按照 Task1 -> Task4 的顺序输出日志
    sorted_keys = sorted(task_files.keys())

    for task_name in sorted_keys:
        ids = task_files[task_name]
        # 随机打乱顺序，有助于训练
        random.shuffle(ids)

        out_path = os.path.join(OUTPUT_DIR, f"{task_name}.txt")
        with open(out_path, "w") as f:
            f.write("\n".join(ids))
        print(f"  [Task]  {task_name}.txt: {len(ids)} 张图片")

    print(f"\n全部完成！文件已保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
