import torch
import numpy as np
import json
import os
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 属性源文件 (上一步 LLM 生成的 JSON 文件路径)
# 里面应该包含 33 个类，每个类 25 句话描述
ATTRIBUTES_FILE = r"/home/gdut-627/huangjiayu/OW_OVD-master/uav_attributes3.json"

# 2. 输出目录 (建议直接放在 SOWOD_Merged_VOC 下面，方便统一管理)
OUTPUT_ROOT = r"/home/gdut-627/106G/public-dataset/OWOD/UAV-OWD/SOWOD_Merged_VOC/SOWOD_Split"
TEXT_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, "texts")

# 3. 模型配置
MODEL_ID = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= SOWOD 类别定义 (优化的 10-8-7-8 划分) =================
# 必须与 vis2coco1.py 里的 SOWOD_TASKS 完全一致，严格按顺序！

T1_CLASSES = [
    'car', 'truck', 'bus', 'van', 'pedestrian', 'people',
    'bicycle', 'motor', 'tricycle', 'awning-tricycle'
]

T2_CLASSES = [
    'airplane', 'helicopter', 'ship', 'harbor',
    'bridge', 'storage-tank', 'chimney', 'dam'
]

T3_CLASSES = [
    'airport', 'train-station', 'overpass', 'toll-station',
    'service-area', 'roundabout', 'windmill'
]

T4_CLASSES = [
    'baseball-field', 'basketball-court', 'tennis-court',
    'ground-track-field', 'soccer-ball-field', 'swimming-pool',
    'stadium', 'golf-field'
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
        raise FileNotFoundError(f"找不到 {ATTRIBUTES_FILE}。请检查路径或先运行 LLM 脚本生成该文件！")

    print(f"Reading attributes from {ATTRIBUTES_FILE}...")
    with open(ATTRIBUTES_FILE, 'r', encoding='utf-8') as f:
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
        print(f"Generating GT embeddings for {len(cumulative_classes)} classes...")

        # 核心修复: 将连字符(-)替换为空格，让 CLIP 更好地理解语义
        # e.g., "storage-tank" -> "storage tank"
        clean_class_names = [c.replace("-", " ") for c in cumulative_classes]

        gt_embeds = encode_texts(model, processor, clean_class_names, desc=f"T{task_id} GT Embeds").numpy()

        npy_path = os.path.join(OUTPUT_ROOT, f"t{task_id}_gt_embeddings.npy")
        np.save(npy_path, gt_embeds)
        print(f"Saved GT Embeddings: {npy_path} {gt_embeds.shape}")

        # --- C. 处理属性 (Attributes) ---
        print(f"Processing attributes for new classes in Task {task_id}...")

        current_task_att_texts = []

        for cls_name in current_classes:
            if cls_name in attributes_data:
                attrs = attributes_data[cls_name]
                # 截取前 25 条，确保维度统一
                if len(attrs) > 25:
                    attrs = attrs[:25]
                elif len(attrs) < 25:
                    print(
                        f"⚠️ Warning: '{cls_name}' has only {len(attrs)} attributes (expected 25). Padding with class name.")
                    while len(attrs) < 25:
                        attrs.append(f"An aerial photo of a {cls_name.replace('-', ' ')}")

                current_task_att_texts.extend(attrs)
                all_class_texts_mapping[cls_name] = attrs
            else:
                print(f"❌ Error: No attributes found for '{cls_name}' in JSON! Using prompts backup.")
                backup_attrs = [f"An aerial photo of a {cls_name.replace('-', ' ')}"] * 25
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
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_class_texts_mapping, f, indent=4)
    print(f"Saved class_texts.json with {len(all_class_texts_mapping)} classes.")

    print("\n✅ Process Completed Successfully!")
    print(f"Please check {OUTPUT_ROOT} for .npy and .pth files.")


if __name__ == "__main__":
    main()