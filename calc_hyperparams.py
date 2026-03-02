import json
import numpy as np

def analyze_dataset_and_recommend(json_file_path):
    print(f"正在读取标注文件: {json_file_path} ...")
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    annotations = data.get('annotations', [])
    if not annotations:
        print("未找到标注信息，请检查 JSON 文件格式！")
        return

    areas = []
    widths = []
    heights = []

    for ann in annotations:
        # COCO 格式的 bbox 是 [x_min, y_min, width, height]
        bbox = ann.get('bbox', [])
        if len(bbox) == 4:
            w, h = bbox[2], bbox[3]
            if w > 0 and h > 0:
                widths.append(w)
                heights.append(h)
                areas.append(w * h)

    if not areas:
        print("没有有效的边界框数据！")
        return

    areas = np.array(areas)
    widths = np.array(widths)
    heights = np.array(heights)

    # 基础统计信息
    avg_area = np.mean(areas)
    median_area = np.median(areas)
    avg_w = np.mean(widths)
    avg_h = np.mean(heights)
    avg_size = (avg_w + avg_h) / 2.0

    # 统计小目标比例 (按照 COCO 标准，area < 32^2 = 1024 算作小目标)
    small_objects = np.sum(areas < 1024)
    small_ratio = small_objects / len(areas) * 100

    print("\n" + "="*40)
    print("🎯 数据集基础统计信息")
    print("="*40)
    print(f"总目标数: {len(areas)}")
    print(f"平均宽度: {avg_w:.2f} 像素")
    print(f"平均高度: {avg_h:.2f} 像素")
    print(f"平均绝对尺寸 ((w+h)/2): {avg_size:.2f} 像素")
    print(f"平均面积: {avg_area:.2f} 平方像素")
    print(f"中位数面积: {median_area:.2f} 平方像素")
    print(f"COCO标准小目标比例 (面积<1024): {small_ratio:.2f}%")

    print("\n" + "="*40)
    print("💡 超参数修改建议 (针对 uav_d_head.py)")
    print("="*40)

    # 1. 推荐 tau_scale
    # 在 uav_d_head.py 中: w_small = exp(-area / tau_scale)
    # 当 area = tau_scale 时，w_small = exp(-1) ≈ 0.36
    # 为了让比“平均大小”小的目标获得更大的 NWD 权重，tau_scale 建议设为平均面积或中位数面积
    recommended_tau = avg_area if avg_area < 2000 else median_area
    print(f"1. tau_scale (难度感知尺度):")
    print(f"   当前代码硬编码值 : 900")
    print(f"   💡 推荐修改为  : {int(recommended_tau)}")
    print(f"   解释: 当目标面积等于 {int(recommended_tau)} 时，NWD 与 IoU 的混合权重将各占一半左右。")

    # 2. 推荐 nwd_constant
    # 原论文中 AI-TOD 数据集平均尺寸约为 12.8 像素，因此 C=12.8
    # 我们应当根据你自己的数据集平均尺寸来设置
    recommended_nwd_c = avg_size
    print(f"\n2. nwd_constant (NWD 常数分母):")
    print(f"   当前代码硬编码值 : 12.8 (这是 AI-TOD 的默认值)")
    print(f"   💡 推荐修改为  : {recommended_nwd_c:.1f}")
    print(f"   解释: 该常数用于将 Wasserstein 距离映射到 0~1 之间，建议设置为数据集目标的平均绝对尺寸。")

    # 3. 推荐 inner_ratio
    print(f"\n3. inner_ratio (Inner-WIoU 缩放比):")
    if small_ratio > 50:
        rec_inner = 0.75
        explanation = "你的数据集中小目标非常多（>50%），更小的 ratio 有助于聚焦核心特征。"
    else:
        rec_inner = 0.85
        explanation = "你的数据集目标尺寸较为均衡，维持 0.85 比较合适。"
    print(f"   当前代码默认值 : 0.85")
    print(f"   💡 推荐修改为  : {rec_inner}")
    print(f"   解释: {explanation}")

if __name__ == "__main__":
    # 请将这里的路径替换为你 base.py 里的实际路径
    JSON_PATH = 'D:\\UAV-OWD\\SOWOD_Merged_VOC\\COCO_JSONB\\instances_train_t1.json'
    analyze_dataset_and_recommend(JSON_PATH)