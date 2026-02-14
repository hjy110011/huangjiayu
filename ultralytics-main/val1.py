from ultralytics import YOLO
import os

# --- 配置 ---
MODEL_PATH = '/home/gdut-627/huangjiayu/ultralytics-main/runs/Yolo-worldv2n2/weights/best.pt'
IMAGE_DIR = '/home/gdut-627/huangjiayu/datasets/val_500/val/images'
PROJECT_NAME = '/home/gdut-627/huangjiayu/ultralytics-main/runs/val_selected_classes'

# 你想评估的类别（零样本）
SELECTED_CLASSES = ['people','traffic-sign',
    'boat', 'traffic-light', 'ship',
    'tricycle', 'bridge']

# --- 检查文件 ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"未找到模型: {MODEL_PATH}")
if not os.path.exists(IMAGE_DIR):
    raise FileNotFoundError(f"未找到图片目录: {IMAGE_DIR}")

# --- 加载模型 ---
model = YOLO(MODEL_PATH)

print(f"开始在 CODrone 零样本数据集上评估类别: {SELECTED_CLASSES}")

# --- 使用 predict 方法进行零样本类别评估 ---
results = model.predict(
    source=IMAGE_DIR,       # 图像目录
    classes=SELECTED_CLASSES,  # 只评估指定类别
    imgsz=640,
    conf=0.001,
    save=True,               # 保存预测结果
    project=PROJECT_NAME,
    name='run_selected_classes'
)

print("评估完成，结果已保存。")
