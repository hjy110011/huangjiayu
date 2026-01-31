import torch
import numpy as np
import json
import os
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# ================= 配置区域 =================
# 输出根目录 (根据您的配置文件路径调整)
OUTPUT_ROOT = "data/VOC2007/SOWOD"
TEXT_OUTPUT_DIR = "data/texts/SOWOD"

# 模型 ID (YOLO-World 常用基座)
MODEL_ID = "openai/clip-vit-base-patch32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 属性源文件 (请确保路径正确)
ATTRIBUTES_FILE = "/home/gdut-627/huangjiayu/OW_OVD-master/attributes_sowod22.json"

# ================= 类别定义 (S-OWODB Task Splits) =================
# Task 1 (VOC 20 classes)
T1_CLASSES = [
    "airplane", "bicycle", "bird", "boat", "bus", "car",
    "cat", "cow", "dog", "horse", "motorcycle", "sheep",
    "train", "elephant", "bear", "zebra", "giraffe", "truck", "person"
]
# Task 2 (20 classes)
T2_CLASSES = [
    "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "chair", "dining table",
    "potted plant", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "microwave", "oven", "toaster", "sink", "refrigerator", "bed", "toilet", "couch"
]
# Task 3 (20 classes)
T3_CLASSES = [
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake"
]
# Task 4 (20 classes)
T4_CLASSES = [
    "laptop", "mouse", "remote", "keyboard", "cell phone",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "tv", "bottle"
]

TASKS = [T1_CLASSES, T2_CLASSES, T3_CLASSES, T4_CLASSES]


# ================= 工具函数 =================

def setup_directories():
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)
    if not os.path.exists(TEXT_OUTPUT_DIR):
        os.makedirs(TEXT_OUTPUT_DIR)
    print(f"Directories created: \n  {OUTPUT_ROOT}\n  {TEXT_OUTPUT_DIR}")


def load_model():
    print(f"Loading CLIP model: {MODEL_ID} on {DEVICE}...")
    model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model.eval()
    return model, processor


def encode_texts(model, processor, text_list, batch_size=32, desc="Encoding"):
    all_embeds = []
    # 添加进度条
    for i in tqdm(range(0, len(text_list), batch_size), desc=desc):
        batch_texts = text_list[i: i + batch_size]
        inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
        # 归一化 (YOLO-World 核心要求)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        all_embeds.append(text_features.cpu())

    if len(all_embeds) > 0:
        return torch.cat(all_embeds, dim=0)
    return torch.empty(0, 512)


# ================= 主逻辑 =================

def main():
    setup_directories()

    # 1. 加载属性数据
    if not os.path.exists(ATTRIBUTES_FILE):
        raise FileNotFoundError(
            f"Missing {ATTRIBUTES_FILE}. Please save the JSON from the previous step as 'attributes.json'.")

    with open(ATTRIBUTES_FILE, 'r') as f:
        attributes_data = json.load(f)
    print(f"Loaded attributes for {len(attributes_data)} classes.")

    # 2. 加载模型
    model, processor = load_model()

    # 3. 逐任务生成文件
    cumulative_classes = []
    cumulative_att_texts = []

    # 用于 class_texts.json
    all_class_texts_mapping = {}

    for task_idx, current_classes in enumerate(TASKS):
        task_id = task_idx + 1
        print(f"\n================ Processing Task {task_id} ================")

        # --- A. 更新累积类别 ---
        cumulative_classes.extend(current_classes)

        # --- B. 生成 GT Class Embeddings (.npy) ---
        print(f"Generating GT embeddings for {len(cumulative_classes)} classes...")
        # 这里的 prompt 策略可以是类名本身，也可以是 "a photo of {class}"
        gt_embeds = encode_texts(model, processor, cumulative_classes, desc=f"Encoding T{task_id} GT").numpy()

        npy_filename = f"t{task_id}_gt_embeddings.npy"
        npy_path = os.path.join(OUTPUT_ROOT, npy_filename)
        np.save(npy_path, gt_embeds)
        print(f"Saved GT Embeddings: {npy_path} {gt_embeds.shape}")

        # --- C. 处理属性嵌入 (.pth) ---
        print(f"Processing attributes for new classes in Task {task_id}...")

        # 获取当前任务新类别的属性文本
        current_task_att_texts = []
        for cls_name in current_classes:
            if cls_name in attributes_data:
                # 确保每个类正好 25 个属性 (论文设定)
                attrs = attributes_data[cls_name][:25]
                if len(attrs) < 25:
                    print(f"Warning: Class {cls_name} has fewer than 25 attributes!")
                current_task_att_texts.extend(attrs)
            else:
                print(f"Error: Attributes for class '{cls_name}' not found in JSON!")
                # 填充空属性以防崩溃 (实际应报错停止)
                current_task_att_texts.extend([f"{cls_name} attribute"] * 25)

        # 更新累积属性文本列表
        cumulative_att_texts.extend(current_task_att_texts)

        # 编码所有累积属性
        print(f"Encoding cumulative attributes ({len(cumulative_att_texts)} texts)...")
        cumulative_att_embeds = encode_texts(model, processor, cumulative_att_texts,
                                             desc=f"Encoding T{task_id} Attributes")

        # 构建 .pth 数据结构
        att_data = {
            'att_text': cumulative_att_texts,
            'att_embedding': cumulative_att_embeds
        }

        # 构建文件名：task_att_1_2_3_4_embeddings.pth
        task_suffix = "_".join([str(i + 1) for i in range(task_id)])
        pth_filename = f"task_att_{task_suffix}_embeddings.pth"
        pth_path = os.path.join(OUTPUT_ROOT, pth_filename)

        torch.save(att_data, pth_path)
        print(f"Saved Attribute Embeddings: {pth_path}")
        print(f"  - Shape: {cumulative_att_embeds.shape}")

        # --- D. 收集 class_texts.json 数据 (已修改) ---
        # 核心修改：从 attributes_data 中读取详细描述，而不是仅使用类名
        for cls_name in current_classes:
            if cls_name in attributes_data:
                # 如果 attributes.json 中有定义，直接使用那 25 条描述
                all_class_texts_mapping[cls_name] = attributes_data[cls_name]
            else:
                # 兜底：如果没有定义，退回到只使用类名 (防止报错)
                print(f"Warning: Attributes not found for {cls_name}, using class name only.")
                all_class_texts_mapping[cls_name] = [cls_name]

    # --- E. 生成 class_texts.json ---
    # 这个文件包含所有 80 个类别的映射
    print(f"\nGenerating {TEXT_OUTPUT_DIR}/class_texts.json...")
    json_path = os.path.join(TEXT_OUTPUT_DIR, "class_texts.json")
    with open(json_path, 'w') as f:
        json.dump(all_class_texts_mapping, f, indent=4)
    print(f"Saved class_texts.json with {len(all_class_texts_mapping)} entries.")

    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()