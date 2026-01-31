import random

import cv2
import numpy as np
import os

# --- 配置部分 ---
# 请使用您在上一轮生成的 200 张图片的输出路径
OUTPUT_ROOT = 'D:\\huangjiayu\\datasets\\val_200'
IMAGE_DIR = os.path.join(OUTPUT_ROOT, 'images')
LABEL_DIR = os.path.join(OUTPUT_ROOT, 'labels')
VISUAL_OUTPUT_DIR = os.path.join(OUTPUT_ROOT, 'visual_check')

# 在上一步骤中定义的类别映射 (必须保持一致!)
COD_CLASSES = [
    'car', 'bus', 'truck', 'van', 'pedestrian', 'bicycle', 'tricycle',
    'awning-tricycle', 'motor', 'boat', 'traffic-sign', 'traffic-light'
]
# ------------------

# 确保可视化输出目录存在
os.makedirs(VISUAL_OUTPUT_DIR, exist_ok=True)


def visualize_yolo_labels(image_file):
    """
    读取图片和对应的 YOLO 标签，将 HBB 绘制在图片上。
    """
    base_name = os.path.splitext(image_file)[0]
    img_path = os.path.join(IMAGE_DIR, image_file)
    label_path = os.path.join(LABEL_DIR, base_name + '.txt')
    output_path = os.path.join(VISUAL_OUTPUT_DIR, 'CHECK_' + image_file)

    if not os.path.exists(label_path):
        print(f"标签文件不存在: {label_path}")
        return

    # 1. 读取图片
    img = cv2.imread(img_path)
    if img is None:
        print(f"无法读取图片: {img_path}")
        return

    H, W, _ = img.shape

    # 2. 读取 YOLO 标签
    with open(label_path, 'r') as f:
        labels = f.readlines()

    # 3. 遍历标签并绘制边界框
    for line in labels:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:])

        # 将归一化坐标转换回像素坐标
        xc = int(x_center * W)
        yc = int(y_center * H)
        w = int(width * W)
        h = int(height * H)

        # 计算矩形的左上角 (xmin, ymin) 和右下角 (xmax, ymax) 坐标
        xmin = int(xc - w / 2)
        ymin = int(yc - h / 2)
        xmax = int(xc + w / 2)
        ymax = int(yc + h / 2)

        # 确定类别名称 (用于显示)
        class_name = COD_CLASSES[class_id] if class_id < len(COD_CLASSES) else f"Class_{class_id}"

        # 绘制边界框 (绿色，线宽 2)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # 绘制类别标签
        cv2.putText(img, class_name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 4. 保存可视化结果
    cv2.imwrite(output_path, img)
    print(f"已生成检查图片: {output_path}")


# 随机选择几张图片进行检查
all_images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
check_samples = random.sample(all_images, min(100, len(all_images)))  # 检查 5 张或更少

for img_file in check_samples:
    visualize_yolo_labels(img_file)

print(f"\n请检查 {VISUAL_OUTPUT_DIR} 文件夹中的图片，确认边界框是否正确包围目标。")