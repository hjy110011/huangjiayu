from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model_yaml = r"/home/gdut-627/huangjiayu/ultralytics-main/runs/detect/train5/weights/last.pt"
    data_yaml = r"/home/gdut-627/huangjiayu/datasets/VisDrone11/VisDrone_data.yaml"
    # model = YOLO(model_yaml, task='detect').load(pre_model)  # build from YAML and transfer weights
    model = YOLO(model_yaml, task='detect')
    model.info()
    # Train the model
    # results = model.train(data=data_yaml, epochs=150, batch=16)
    results = model.train(resume = True)
