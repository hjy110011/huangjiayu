import torch
import numpy as np
import json
import os
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 属性源文件 (上一步 LLM 生成的文件)
ATTRIBUTES_FILE = "/home/gdut-627/huangjiayu/OW_OVD-master/xview_attributes1.json"

# 2. 输出目录 (必须与之前的步骤一致: xView_VOC/SOWOD_User_Split)
OUTPUT_ROOT = "/home/gdut-627/106G/public-dataset/OWOD/xview/xView_VOC/SOWOD_User_Split"
TEXT_OUTPUT_DIR = "data/texts/SOWOD"  # 文本映射文件存放位置

# 3. 模型 ID
MODEL_ID = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 类别定义 (xView 15-15-15-15) =================
# 注意：必须与之前的 JSON 生成脚本完全一致
# 这里的名字带有下划线，用于文件查找；输入 CLIP 时我们会把下划线换成空格。

T1_CLASSES = [
    "Fixed-wing_Aircraft", "Small_Aircraft", "Cargo_Plane", "Helicopter",
    "Passenger_Vehicle", "Small_Car", "Bus", "Pickup_Truck", "Truck",
    "Railway_Vehicle", "Passenger_Car", "Locomotive", "Maritime_Vessel",
    "Motorboat", "Fishing_Vessel"
]

T2_CLASSES = [
    "Utility_Truck", "Cargo_Truck", "Truck_w_Box", "Truck_Tractor",
    "Trailer", "Truck_w_Flatbed", "Truck_w_Liquid", "Crane_Truck",
    "Ferry", "Yacht", "Container_Ship", "Oil_Tanker", "Engineering_Vehicle",
    "Dump_Truck", "Haul_Truck"
]

T3_CLASSES = [
    "Tower_crane", "Container_Crane", "Reach_Stacker", "Straddle_Carrier",
    "Mobile_Crane", "Scraper_Tractor", "Front_loader_Bulldozer", "Excavator",
    "Cement_Mixer", "Ground_Grader", "Hut_Tent", "Shed", "Construction_Site",
    "Vehicle_Lot", "Helipad"
]

T4_CLASSES = [
    "Building", "Aircraft_Hangar", "Damaged_Building", "Facility",
    "Storage_Tank", "Shipping_container_lot", "Shipping_Container", "Pylon",
    "Tower", "Flat_Car", "Tank_car", "Cargo_Car", "Sailboat", "Tugboat", "Barge"
]

TASKS = [T1_CLASSES, T2_CLASSES, T3_CLASSES, T4_CLASSES]


# ================= 工具函数 =================

def setup_directories():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(TEXT_OUTPUT_DIR, exist_ok=True)
    print(f"Directories ready:\n  Data: {OUTPUT_ROOT}\n  Texts: {TEXT_OUTPUT_DIR}")


def load_model():
    print(f"Loading CLIP model: {MODEL_ID} on {DEVICE}...")
    model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model.eval()
    return model, processor


def encode_texts(model, processor, text_list, batch_size=32, desc="Encoding"):
    all_embeds = []
    # 简单的截断处理 (max_length=77 是 CLIP 的限制)
    for i in tqdm(range(0, len(text_list), batch_size), desc=desc):
        batch_texts = text_list[i: i + batch_size]
        inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(
            DEVICE)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
        # 归一化 (Normalization) - 这一步对 CLIP 特征匹配至关重要
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        all_embeds.append(text_features.cpu())

    if len(all_embeds) > 0:
        return torch.cat(all_embeds, dim=0)
    return torch.empty(0, 512)


# ================= 主逻辑 =================

def main():
    setup_directories()

    # 1. 加载属性文件
    if not os.path.exists(ATTRIBUTES_FILE):
        raise FileNotFoundError(f"找不到 {ATTRIBUTES_FILE}。请先运行上一步的 LLM 脚本生成该文件！")

    print(f"Reading attributes from {ATTRIBUTES_FILE}...")
    with open(ATTRIBUTES_FILE, 'r') as f:
        attributes_data = json.load(f)

    # 2. 加载模型
    model, processor = load_model()

    # 3. 逐任务生成
    cumulative_classes = []  # 累积的类别名 (用于生成 GT Embeddings)
    cumulative_att_texts = []  # 累积的属性文本 (用于生成 Attributes Embeddings)
    all_class_texts_mapping = {}  # 用于保存 class_texts.json

    for task_idx, current_classes in enumerate(TASKS):
        task_id = task_idx + 1
        print(f"\n================ Processing Task {task_id} ================")

        # --- A. 更新累积类别 ---
        cumulative_classes.extend(current_classes)

        # --- B. 生成 GT Class Embeddings (.npy) ---
        # 这里的逻辑是：生成当前任务及之前所有类别的 CLIP 向量
        # 关键修改：将下划线替换为空格，让 CLIP 更好地理解语义
        # e.g., "Fixed-wing_Aircraft" -> "Fixed-wing Aircraft"
        print(f"Generating GT embeddings for {len(cumulative_classes)} classes...")
        clean_class_names = [c.replace("_", " ") for c in cumulative_classes]

        gt_embeds = encode_texts(model, processor, clean_class_names, desc=f"T{task_id} GT Embeds").numpy()

        npy_path = os.path.join(OUTPUT_ROOT, f"t{task_id}_gt_embeddings.npy")
        np.save(npy_path, gt_embeds)
        print(f"Saved GT Embeddings: {npy_path} {gt_embeds.shape}")

        # --- C. 处理属性 (Attributes) ---
        print(f"Processing attributes for new classes in Task {task_id}...")

        current_task_att_texts = []

        for cls_name in current_classes:
            # 尝试获取属性，如果 JSON 里没有，就打印警告并用类名兜底
            # 注意：attributes_data 的 key 可能是带下划线的，也可能是原本的
            # 我们在上一步脚本里是按 cls_name (带下划线) 保存的

            if cls_name in attributes_data:
                attrs = attributes_data[cls_name]
                # 截取前 25 条，确保维度统一
                if len(attrs) > 25:
                    attrs = attrs[:25]
                elif len(attrs) < 25:
                    print(
                        f"⚠️ Warning: {cls_name} has only {len(attrs)} attributes (expected 25). Padding with class name.")
                    while len(attrs) < 25:
                        attrs.append(f"A photo of {cls_name.replace('_', ' ')}")

                current_task_att_texts.extend(attrs)
                all_class_texts_mapping[cls_name] = attrs
            else:
                print(f"❌ Error: No attributes found for '{cls_name}'! Using prompts backup.")
                backup_attrs = [f"A satellite image of {cls_name.replace('_', ' ')}"] * 25
                current_task_att_texts.extend(backup_attrs)
                all_class_texts_mapping[cls_name] = backup_attrs

        # 更新累积属性
        cumulative_att_texts.extend(current_task_att_texts)

        # 编码属性 (.pth)
        print(f"Encoding cumulative attributes ({len(cumulative_att_texts)} lines)...")
        cumulative_att_embeds = encode_texts(model, processor, cumulative_att_texts, desc=f"T{task_id} Att Embeds")

        att_data = {
            'att_text': cumulative_att_texts,
            'att_embedding': cumulative_att_embeds
        }

        # 文件命名: task_att_1_embeddings.pth, task_att_1_2_embeddings.pth
        task_suffix = "_".join([str(i + 1) for i in range(task_id)])
        pth_path = os.path.join(OUTPUT_ROOT, f"task_att_{task_suffix}_embeddings.pth")

        torch.save(att_data, pth_path)
        print(f"Saved Attributes: {pth_path} {cumulative_att_embeds.shape}")

    # --- D. 生成 class_texts.json ---
    print(f"\nGenerating {TEXT_OUTPUT_DIR}/class_texts.json...")
    json_path = os.path.join(TEXT_OUTPUT_DIR, "class_texts.json")
    with open(json_path, 'w') as f:
        json.dump(all_class_texts_mapping, f, indent=4)
    print(f"Saved class_texts.json with {len(all_class_texts_mapping)} classes.")

    print("\n✅ Process Completed Successfully!")
    print(f"Please check {OUTPUT_ROOT} for .npy and .pth files.")


if __name__ == "__main__":
    main()