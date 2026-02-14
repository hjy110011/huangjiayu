from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model_yaml = r"/home/gdut-627/huangjiayu/ultralytics-main/ultralytics/cfg/models/11/MASF-YOLO.yaml"
    data_yaml = r"/home/gdut-627/huangjiayu/datasets/VisDrone11/VisDrone_data.yaml"
    pre_model = r"/home/gdut-627/huangjiayu/ultralytics-main/yolo11s.pt"
    resume_model = r"/home/gdut-627/huangjiayu/ultralytics-main/runs/detect/yolo11s/weights/last.pt"

    # model = YOLO(model_yaml, task='detect').load(pre_model)  # build from YAML and transfer weights
    # model = YOLO(model_yaml, task='detect')
    # Train the model
    # model = YOLO(resume_model).reset_scale("n")
    # results = model.train(data=data_yaml, epochs=150, batch=16)
    # results = model.train(epochs=175, resume=True)
    model = YOLO(model_yaml, task='detect').reset_scale("n")
    model.info()
    # Train the model
    # results = model.train(data=data_yaml, epochs=150, batch=16)
    results = model.train(data=data_yaml, epochs=150, pretrained=False, batch=4)