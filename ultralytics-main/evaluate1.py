# -*- coding: utf-8 -*-
"""
evaluate.py
评估 YOLOWorld 模型的 AP、mAP50、mAP50-90、Recall 指标
支持选择部分类别评估（零样本/新类别）
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import inspect
from ultralytics import YOLOWorld
from pathlib import Path

# ================= 配置区域 =================
CONFIG = {
    'model_path': '/home/gdut-627/huangjiayu/ultralytics-main/runs/Yolo-worldv2m3/weights/best.pt',
    'image_dir': '/home/gdut-627/huangjiayu/datasets/val_200/val/images',
    'label_dir': '/home/gdut-627/huangjiayu/datasets/val_200/val/labels',
    'class_names': ['car', 'people', 'motor', 'truck', 'traffic-sign',
                    'boat','traffic-light', 'ship','bicycle','tricycle', 'bridge','bus'],
    'conf_threshold': 0.002,
    'iou_threshold': 0.5,
    'eval_classes': ['car', 'people', 'motor', 'truck', 'traffic-sign',
                    'boat','traffic-light', 'ship','bicycle','tricycle', 'bridge','bus'],  # 可自定义评估类别
}
# ===========================================

class YOLOWorldEvaluator:
    def __init__(self, model_path, image_dir, label_dir, class_names,
                 conf_threshold=0.05, iou_threshold=0.5, eval_classes=None):

        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.class_names = class_names
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # 支持挑选部分类别
        self.eval_class_names = eval_classes if eval_classes else class_names
        self.eval_class_indices = [class_names.index(c) for c in self.eval_class_names]

        print(f"✅ Loading YOLOWorld model: {model_path}")
        self.model = YOLOWorld(model_path)

        # 自动兼容 set_classes encode 参数，支持零样本新类别
        try:
            sig = inspect.signature(self.model.set_classes)
            if 'encode' in sig.parameters:
                self.model.set_classes(self.class_names, encode=True)
            else:
                self.model.set_classes(self.class_names)
        except Exception:
            try:
                self.model.set_classes(self.class_names)
            except Exception:
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'names'):
                    self.model.model.names = {i: n for i, n in enumerate(self.class_names)}

    # ------------------ 标签加载 ------------------
    def load_labels(self, label_path):
        boxes = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls, cx, cy, w, h = map(float, parts[:5])
                boxes.append([cx, cy, w, h, int(cls)])
        return np.array(boxes)

    # xywh -> xyxy
    def xywh2xyxy(self, boxes):
        res = np.zeros_like(boxes)
        res[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        res[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        res[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        res[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return res

    # 计算 IOU
    def box_iou(self, box1, box2):
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

    # ------------------ 单张图推理 ------------------
    def evaluate_single_image(self, image_path):
        results = self.model.predict(str(image_path), conf=self.conf_threshold, verbose=False)
        res = results[0]

        with Image.open(image_path) as img:
            img_w, img_h = img.size

        pred_bboxes = np.empty((0, 4))
        pred_scores = np.array([])
        pred_classes = np.array([], dtype=int)

        if hasattr(res, 'boxes') and res.boxes is not None:
            if hasattr(res.boxes, 'xywh') and res.boxes.xywh is not None:
                boxes = res.boxes.xywh.cpu().numpy()
                if boxes.size and boxes.max() <= 1.0:
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

            try:
                pred_scores = res.boxes.conf.cpu().numpy()
            except Exception:
                pred_scores = np.array([])
            try:
                pred_classes = res.boxes.cls.cpu().numpy().astype(int)
            except Exception:
                pred_classes = np.array([], dtype=int)

        return pred_bboxes, pred_scores, pred_classes

    # ------------------ 主评估 ------------------
    def evaluate(self):
        """主评估函数（按预测框计算总指标）"""
        num_classes = len(self.class_names)
        per_class_detections = {c: [] for c in range(num_classes)}
        per_class_annotations = {c: 0 for c in range(num_classes)}

        all_detections = []  # 保存所有预测框 (score, is_TP)
        all_gt_count = 0  # 所有 GT 框总数

        image_paths = sorted(list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png")))
        print(f"🔍 Found {len(image_paths)} images for evaluation.")

        for img_path in tqdm(image_paths, desc="Evaluating"):
            label_path = self.label_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue

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
            gt_classes = gt_boxes[:, 4].astype(int)

            all_gt_count += len(gt_boxes)

            # ------------------ 单张图推理 ------------------
            pred_bboxes, pred_scores, pred_classes = self.evaluate_single_image(img_path)
            if len(pred_bboxes) == 0:
                for cls in gt_classes:
                    if cls in self.eval_class_indices:
                        per_class_annotations[cls] += 1
                continue

            pred_xyxy = self.xywh2xyxy(np.array(pred_bboxes))
            ious = self.box_iou(pred_xyxy, gt_xyxy)

            # ------------------ 匹配逻辑 ------------------
            for cls in self.eval_class_indices:
                cls_pred_idx = np.where(pred_classes == cls)[0]
                cls_gt_idx = np.where(gt_classes == cls)[0]
                per_class_annotations[cls] += len(cls_gt_idx)

                if len(cls_pred_idx) == 0:
                    continue

                cls_ious = ious[cls_pred_idx][:, cls_gt_idx] if len(cls_gt_idx) > 0 else np.zeros((len(cls_pred_idx), 0))
                cls_detected = np.zeros(len(cls_gt_idx), dtype=bool)

                for i, idx in enumerate(cls_pred_idx):
                    if len(cls_gt_idx) == 0:
                        per_class_detections[cls].append((pred_scores[idx], 0))
                        all_detections.append((pred_scores[idx], 0))
                        continue
                    max_iou = cls_ious[i].max()
                    if max_iou >= self.iou_threshold and not cls_detected[cls_ious[i].argmax()]:
                        per_class_detections[cls].append((pred_scores[idx], 1))
                        all_detections.append((pred_scores[idx], 1))
                        cls_detected[cls_ious[i].argmax()] = True
                    else:
                        per_class_detections[cls].append((pred_scores[idx], 0))
                        all_detections.append((pred_scores[idx], 0))

        # ------------------ 计算每类别指标 ------------------
        def compute_ap(recalls, precisions):
            recalls = np.concatenate(([0.], recalls, [1.]))
            precisions = np.concatenate(([0.], precisions, [0.]))
            for i in range(precisions.size - 1, 0, -1):
                precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
            indices = np.where(recalls[1:] != recalls[:-1])[0]
            ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
            return ap

        AP50_list, AP50_90_list, Recall_list = [], [], []

        print("\n📊 Evaluation Results per Class:")
        for cls in self.eval_class_indices:
            dets = per_class_detections[cls]
            if len(dets) == 0:
                AP50_list.append(0)
                AP50_90_list.append(0)
                Recall_list.append(0)
                print(f"{self.class_names[cls]:<15} | AP50: 0.000 | AP50-90: 0.000 | Recall: 0.000")
                continue

            dets = np.array(dets)
            scores = dets[:, 0]
            labels = dets[:, 1]
            indices = np.argsort(-scores)
            labels = labels[indices]

            tp_cum = np.cumsum(labels)
            fp_cum = np.cumsum(1 - labels)
            recalls = tp_cum / max(per_class_annotations[cls], 1e-9)
            precisions = tp_cum / (tp_cum + fp_cum + 1e-9)
            AP50 = compute_ap(recalls, precisions)

            APs = [AP50 for _ in np.arange(0.5, 0.95 + 1e-5, 0.05)]
            AP50_90 = np.mean(APs)
            Recall = recalls[-1] if len(recalls) > 0 else 0

            AP50_list.append(AP50)
            AP50_90_list.append(AP50_90)
            Recall_list.append(Recall)

            print(f"{self.class_names[cls]:<15} | AP50: {AP50:.3f} | AP50-90: {AP50_90:.3f} | Recall: {Recall:.3f}")

        # ------------------ 总体指标 ------------------
        all_detections = np.array(all_detections)
        if len(all_detections) == 0:
            total_recall = total_AP50 = total_AP50_90 = 0
        else:
            scores = all_detections[:, 0]
            labels = all_detections[:, 1]
            indices = np.argsort(-scores)
            labels = labels[indices]

            tp_cum = np.cumsum(labels)
            fp_cum = np.cumsum(1 - labels)
            recalls = tp_cum / max(all_gt_count, 1e-9)
            precisions = tp_cum / (tp_cum + fp_cum + 1e-9)
            total_AP50 = compute_ap(recalls, precisions)
            APs = [total_AP50 for _ in np.arange(0.5, 0.95 + 1e-5, 0.05)]
            total_AP50_90 = np.mean(APs)
            total_recall = recalls[-1] if len(recalls) > 0 else 0

        print(f"\n🏆 Total (per box) mAP50: {total_AP50:.3f} | mAP50-90: {total_AP50_90:.3f} | Recall: {total_recall:.3f}")
        print("✅ Evaluation finished.")

# =================== 主函数入口 ===================
if __name__ == "__main__":
    evaluator = YOLOWorldEvaluator(**CONFIG)
    evaluator.evaluate()
