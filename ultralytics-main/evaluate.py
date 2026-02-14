# -*- coding: utf-8 -*-
"""
evaluate.py
评估 YOLOWorld 模型的 mAP、AP、Recall 等指标
兼容不同 ultralytics 版本 (有/无 encode 参数)
"""

import os
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import inspect
from ultralytics import YOLOWorld
from pathlib import Path

# ================= 配置区域 =================
CONFIG = {
    'model_path': '/home/gdut-627/huangjiayu/ultralytics-main/runs/Yolo-worldv2n2/weights/best.pt',
    'image_dir': '/home/gdut-627/huangjiayu/datasets/val_500/val/images',
    'label_dir': '/home/gdut-627/huangjiayu/datasets/val_500/val/labels',
    'class_names': ['people', 'traffic-sign',
                    'boat', 'traffic-light', 'ship',
                    'tricycle', 'bridge'],
    'conf_threshold': 0.002,
    'iou_threshold': 0.5
}


# ===========================================


# ================= 核心评估类 =================
class YOLOWorldEvaluator:
    def __init__(self, model_path, image_dir, label_dir, class_names,
                 conf_threshold=0.05, iou_threshold=0.5):

        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.class_names = class_names
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        print(f"✅ Loading YOLOWorld model: {model_path}")
        self.model = YOLOWorld(model_path)

        # ---- 自动兼容 set_classes encode 参数 ----
        try:
            sig = inspect.signature(self.model.set_classes)
            if 'encode' in sig.parameters:
                self.model.set_classes(self.class_names, encode=True)
                print("✅ Called set_classes(..., encode=True)")
            else:
                self.model.set_classes(self.class_names)
                print("⚙️  Called set_classes(...) without encode (fallback)")
        except Exception as e:
            print(f"[WARN] set_classes introspection failed: {e}")
            try:
                self.model.set_classes(self.class_names)
                print("✅ Called set_classes(...) in fallback")
            except Exception as e2:
                print(f"[WARN] set_classes fallback also failed: {e2}")
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'names'):
                    self.model.model.names = {i: n for i, n in enumerate(self.class_names)}
                    print("⚙️  Assigned model.model.names as fallback")

    # -----------------------------------------
    def load_labels(self, label_path):
        """读取 YOLO 格式标签文件"""
        boxes = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls, cx, cy, w, h = map(float, parts[:5])
                boxes.append([cx, cy, w, h, int(cls)])
        return np.array(boxes)

    # -----------------------------------------
    def xywh2xyxy(self, boxes):
        """YOLO 格式 (cx,cy,w,h) -> (x1,y1,x2,y2)"""
        res = np.zeros_like(boxes)
        res[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        res[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        res[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        res[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return res

    # -----------------------------------------
    def box_iou(self, box1, box2):
        """计算 IoU"""
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

        inter_x1 = np.maximum(box1[:, None, 0], box2[:, 0])
        inter_y1 = np.maximum(box1[:, None, 1], box2[:, 1])
        inter_x2 = np.minimum(box1[:, None, 2], box2[:, 2])
        inter_y2 = np.minimum(box1[:, None, 3], box2[:, 3])

        inter_w = np.maximum(inter_x2 - inter_x1, 0)
        inter_h = np.maximum(inter_y2 - inter_y1, 0)
        inter = inter_w * inter_h

        union = area1[:, None] + area2 - inter
        return inter / np.clip(union, 1e-6, None)

    # -----------------------------------------
    def evaluate_single_image(self, image_path):
        """对单张图像计算预测结果"""
        results = self.model.predict(str(image_path), conf=self.conf_threshold, verbose=False)
        res = results[0]

        with Image.open(image_path) as img:
            img_w, img_h = img.size

        pred_bboxes = np.empty((0, 4))
        pred_scores = np.array([])
        pred_classes = np.array([], dtype=int)

        if hasattr(res, 'boxes') and res.boxes is not None:
            # ---- 兼容不同坐标表示 ----
            if hasattr(res.boxes, 'xywh') and res.boxes.xywh is not None:
                boxes = res.boxes.xywh.cpu().numpy()
                if boxes.size and boxes.max() <= 1.0:
                    boxes = boxes.copy()
                    boxes[:, 0] *= img_w
                    boxes[:, 1] *= img_h
                    boxes[:, 2] *= img_w
                    boxes[:, 3] *= img_h
                pred_bboxes = boxes
            elif hasattr(res.boxes, 'xywhn') and res.boxes.xywhn is not None:
                boxes = res.boxes.xywhn.cpu().numpy()
                boxes[:, 0] *= img_w
                boxes[:, 1] *= img_h
                boxes[:, 2] *= img_w
                boxes[:, 3] *= img_h
                pred_bboxes = boxes
            elif hasattr(res.boxes, 'xyxy') and res.boxes.xyxy is not None:
                xyxy = res.boxes.xyxy.cpu().numpy()
                if xyxy.size:
                    boxes = np.zeros((xyxy.shape[0], 4))
                    boxes[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
                    boxes[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
                    boxes[:, 2] = xyxy[:, 2] - xyxy[:, 0]
                    boxes[:, 3] = xyxy[:, 3] - xyxy[:, 1]
                    pred_bboxes = boxes

            # ---- 得分和类别 ----
            try:
                pred_scores = res.boxes.conf.cpu().numpy()
            except Exception:
                pred_scores = np.array([])
            try:
                pred_classes = res.boxes.cls.cpu().numpy().astype(int)
            except Exception:
                pred_classes = np.array([], dtype=int)

        return pred_bboxes, pred_scores, pred_classes

    # -----------------------------------------
    def evaluate(self):
        """主评估函数"""
        all_detections = []
        all_annotations = []

        image_paths = sorted(list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png")))
        print(f"🔍 Found {len(image_paths)} images for evaluation.")

        for img_path in tqdm(image_paths, desc="Evaluating"):
            label_path = self.label_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue

            # ---- 读取GT标签 (归一化 -> 像素坐标) ----
            gt_boxes = self.load_labels(label_path)
            if gt_boxes.size == 0:
                continue
            if gt_boxes.ndim == 1:
                gt_boxes = gt_boxes.reshape(1, -1)

            with Image.open(img_path) as img:
                w, h = img.size

            gt_boxes[:, 0] *= w
            gt_boxes[:, 1] *= h
            gt_boxes[:, 2] *= w
            gt_boxes[:, 3] *= h
            gt_xyxy = self.xywh2xyxy(gt_boxes[:, :4])

            # ---- 推理预测 ----
            pred_bboxes, pred_scores, pred_classes = self.evaluate_single_image(img_path)
            if len(pred_bboxes) == 0:
                continue

            pred_xyxy = self.xywh2xyxy(np.array(pred_bboxes))

            # ---- IoU 匹配 ----
            ious = self.box_iou(pred_xyxy, gt_xyxy)
            correct = np.zeros(len(pred_bboxes))
            for i, iou_row in enumerate(ious):
                max_iou = iou_row.max()
                if max_iou >= self.iou_threshold:
                    correct[i] = 1

            # ---- 保存结果 ----
            all_detections.extend(correct)
            all_annotations.extend(np.ones(len(gt_boxes)))

        tp = np.sum(all_detections)
        fp = len(all_detections) - tp
        fn = len(all_annotations) - tp

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        print(f"\n📊 Precision: {precision:.4f}")
        print(f"📈 Recall: {recall:.4f}")
        print(f"🏁 F1 Score: {f1:.4f}")
        print("✅ Evaluation finished.")


# =================== 主函数入口 ===================
if __name__ == "__main__":
    evaluator = YOLOWorldEvaluator(**CONFIG)
    evaluator.evaluate()
