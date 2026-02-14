from ultralytics import YOLO

if __name__ == "__main__":
    # 模型和数据路径
    model_yaml = r"/home/gdut-627/huangjiayu/ultralytics-main/ultralytics/cfg/models/MyModels/MASF-YOLOn.yaml"
    data_yaml = r"/home/gdut-627/huangjiayu/datasets/VisDrone11/VisDrone_data.yaml"

    # 初始化模型
    model = YOLO(model_yaml, task='detect')
    model.info()

    # 指定保存路径
    results = model.train(
        data=data_yaml,
        imgsz=640,
        epochs=150,
        pretrained=False,
        batch=4,
        project="/home/gdut-627/huangjiayu/ultralytics-main/runs",  # 保存的主目录
        name="MASF-YOLOn"  # 保存的子目录名
    )
