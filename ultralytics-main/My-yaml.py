from ultralytics import YOLO

# 加载自定义 yaml
model = YOLO("/home/gdut-627/huangjiayu/ultralytics-main/ultralytics/cfg/models/MyModels/MyYolo1.yaml")

# 打印网络结构和参数量
model.info()

# 生成网络图 (保存为 PNG 文件)
model.plot()  # 默认保存到 runs/models/
