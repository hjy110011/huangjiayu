# from ultralytics import YOLO
# import os
#
# # --- 配置部分 ---
# # 替换为您的模型路径 (VisDrone11 训练的 YOLO-World 权重)
# MODEL_PATH = '/home/gdut-627/huangjiayu/ultralytics-main/runs/Yolo-worldv2n2/weights/best.pt'
#
# # 替换为您创建的 data.yaml 文件路径
# # 该文件应指向您抽取的 200 张 CODrone 图片及其标签
# DATA_YAML_PATH = '/home/gdut-627/huangjiayu/datasets/val_200/data.yaml'
#
# # 结果保存目录 (可选，评估结果将保存在这里)
# PROJECT_NAME = '/home/gdut-627/huangjiayu/ultralytics-main/runs/val'
# # ------------------
#
# # 检查文件是否存在
# if not os.path.exists(MODEL_PATH):
#     raise FileNotFoundError(f"未找到模型文件: {MODEL_PATH}")
# if not os.path.exists(DATA_YAML_PATH):
#     raise FileNotFoundError(f"未找到 data.yaml 文件: {DATA_YAML_PATH}")
#
# print(f"正在加载模型: {MODEL_PATH}")
# # 1. 加载训练好的 YOLO-World 模型
# model = YOLO(MODEL_PATH)
#
# print(f"开始在 CODrone 零样本数据集上运行评估...")
#
# # 2. 运行评估 (Validation/Evaluation)
# # 使用 model.val() 方法进行评估，它会自动读取 data.yaml 中的 'val' 或 'our' 路径和标签
# results = model.val(
#     data=DATA_YAML_PATH,  # 指定配置文件
#     split='val',  # 在 data.yaml 中 'val' 指定的数据集上评估 (即那 200 张图片)
#     imgsz=640,  # 评估时使用的图像尺寸 (应与训练时一致或根据需要调整)
#     conf=0.001,  # 预测置信度阈值 (一般保持默认或较低值以获得召回率)
#     project=PROJECT_NAME,  # 结果保存项目名称
#     name='run1',  # 结果保存实验名称
#     save_json=True,  # 保存 COCO 格式的 JSON 结果文件，便于进一步分析
# )
#
# print("\n--- 评估结果摘要 ---")
#
# # 3. 打印关键评估指标
# if results.box is not None:
#     # mAP50: IoU=0.5 时的平均精度
#     print(f"mAP@0.5 (IoU=50%): {results.box.map50:.4f}")
#
#     # mAP50-95: IoU 从 0.5 到 0.95 (步长 0.05) 的平均 mAP
#     print(f"mAP@0.5:0.95: {results.box.map:.4f}")
#
#     # 召回率 (Recall): 在 IoU=0.5 时的召回率
#     print(f"Recall@0.5: {results.box.recall[0]:.4f}")
#
#     # 精度 (Precision): 在 IoU=0.5 时的精度
#     print(f"Precision@0.5: {results.box.precision[0]:.4f}")
#
#     print("\n详细结果已保存到项目目录。")
# else:
#     print("评估未生成有效的边界框结果。请检查模型、标签和配置文件路径。")



from ultralytics import YOLO
import os

# --- 配置部分 ---
# 替换为您的模型路径 (VisDrone11 训练的 YOLO-World 权重)
MODEL_PATH = '/home/gdut-627/huangjiayu/ultralytics-main/runs/Yolo-worldv2n2/weights/best.pt'

# 替换为您创建的 data.yaml 文件路径
# 该文件应指向您抽取的 200 张 CODrone 图片及其标签
DATA_YAML_PATH = '/home/gdut-627/huangjiayu/datasets/val_500/data.yaml'

# 结果保存目录 (可选，评估结果将保存在这里)
PROJECT_NAME = '/home/gdut-627/huangjiayu/ultralytics-main/runs/val'
# ------------------

# 检查文件是否存在
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"未找到模型文件: {MODEL_PATH}")
if not os.path.exists(DATA_YAML_PATH):
    raise FileNotFoundError(f"未找到 data.yaml 文件: {DATA_YAML_PATH}")

print(f"正在加载模型: {MODEL_PATH}")
# 1. 加载训练好的 YOLO-World 模型
model = YOLO(MODEL_PATH)

print(f"开始在 CODrone 零样本数据集上运行评估...")

# 2. 运行评估 (Validation/Evaluation)
# 使用 model.val() 方法进行评估，它会自动读取 data.yaml 中的 'val' 或 'our' 路径和标签
results = model.val(
    data=DATA_YAML_PATH,  # 指定配置文件
    split='val',  # 在 data.yaml 中 'val' 指定的数据集上评估 (即那 200 张图片)
    imgsz=640,  # 评估时使用的图像尺寸 (应与训练时一致或根据需要调整)
    conf=0.001,  # 预测置信度阈值 (一般保持默认或较低值以获得召回率)
    project=PROJECT_NAME,  # 结果保存项目名称
    name='run1',  # 结果保存实验名称
    save_json=True,  # 保存 COCO 格式的 JSON 结果文件，便于进一步分析
)

print("\n--- 评估结果摘要 ---")

# 3. 打印关键评估指标
if results.box is not None:
    # mAP50: IoU=0.5 时的平均精度
    print(f"mAP@0.5 (IoU=50%): {results.box.map50:.4f}")

    # mAP50-95: IoU 从 0.5 到 0.95 (步长 0.05) 的平均 mAP
    print(f"mAP@0.5:0.95: {results.box.map:.4f}")

    # 召回率 (Recall): 在 IoU=0.5 时的召回率
    print(f"Recall@0.5: {results.box.recall[0]:.4f}")

    # 精度 (Precision): 在 IoU=0.5 时的精度
    print(f"Precision@0.5: {results.box.precision[0]:.4f}")

    print("\n详细结果已保存到项目目录。")
else:
    print("评估未生成有效的边界框结果。请检查模型、标签和配置文件路径。")