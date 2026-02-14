from ultralytics import YOLO

if __name__ == "__main__":
    # Load a model
    model_yaml = r"/home/gdut-627/huangjiayu/ultralytics-main/runs/Yolo-worldv2n-640/weights/best.pt"
    data_yaml = r"/home/gdut-627/huangjiayu/datasets/VisDrone11/VisDrone2019-DET-test-dev/images"



    # Initialize a YOLO-World model
    model = YOLO(model_yaml)  # or choose yolov8m/l-world.pt

    # # Define custom classes
    # model.set_classes(["person", "bus"])

    # Execute prediction for specified categories on an image
    results = model.predict(data_yaml,
                            imgsz=640,
                            project="/home/gdut-627/huangjiayu/ultralytics-main/runs",  # 保存的主目录
                            name="test-640n"
                            )
# yolo detect val model=/home/gdut-627/huangjiayu/ultralytics-main/runs/Yolo-worldv2n-640/weights/best.pt data=/home/gdut-627/huangjiayu/datasets/VisDrone11/VisDrone2019-DET-test-dev/images