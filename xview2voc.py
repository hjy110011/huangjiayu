import os
import json
import shutil
from PIL import Image
from lxml import etree
from tqdm import tqdm

# ================= 配置区域 =================
CONFIG = {
    # xView 原始数据路径
    "geojson_path": "D:\\xview\\xView_train.geojson",  # 下载的 geojson 文件路径
    "images_dir": "D:\\xview\\images\\train_images\\",  # 原始 .tif 图片文件夹路径

    # 输出路径 (会自动创建)
    "output_dir": "D:\\xview\\xView_VOC",

    # 切片参数 (这是保留原图质量的关键)
    "crop_size": 640,  # 切片大小 (例如 608x608, 800x800)
    "overlap": 0.2,  # 切片间的重叠率 (0.2 = 20%)，防止目标被切断
    "min_visibility": 0.8,  # 如果目标被切断，至少保留 60% 才保留标注

    # 是否保存空图片 (没有目标的切片是否保留?)
    "save_empty": False
}

# xView 类别映射 (原始ID -> 类别名)
# 这是一个精简版映射，确保所有ID都有对应的名称
XVIEW_CLASSES = {
    11: 'Fixed-wing Aircraft', 12: 'Small Aircraft', 13: 'Cargo Plane', 15: 'Helicopter',
    17: 'Passenger Vehicle', 18: 'Small Car', 19: 'Bus', 20: 'Pickup Truck',
    21: 'Utility Truck', 23: 'Truck', 24: 'Cargo Truck', 25: 'Truck w/Box',
    26: 'Truck Tractor', 27: 'Trailer', 28: 'Truck w/Flatbed', 29: 'Truck w/Liquid',
    32: 'Crane Truck', 33: 'Railway Vehicle', 34: 'Passenger Car', 35: 'Cargo Car',
    36: 'Flat Car', 37: 'Tank car', 38: 'Locomotive', 40: 'Maritime Vessel',
    41: 'Motorboat', 42: 'Sailboat', 44: 'Tugboat', 45: 'Barge', 47: 'Fishing Vessel',
    49: 'Ferry', 50: 'Yacht', 51: 'Container Ship', 52: 'Oil Tanker',
    53: 'Engineering Vehicle', 54: 'Tower crane', 55: 'Container Crane',
    56: 'Reach Stacker', 57: 'Straddle Carrier', 59: 'Mobile Crane', 60: 'Dump Truck',
    61: 'Haul Truck', 62: 'Scraper/Tractor', 63: 'Front loader/Bulldozer',
    64: 'Excavator', 65: 'Cement Mixer', 66: 'Ground Grader', 71: 'Hut/Tent',
    72: 'Shed', 73: 'Building', 74: 'Aircraft Hangar', 76: 'Damaged Building',
    77: 'Facility', 79: 'Construction Site', 83: 'Vehicle Lot', 84: 'Helipad',
    86: 'Storage Tank', 89: 'Shipping container lot', 91: 'Shipping Container',
    93: 'Pylon', 94: 'Tower'
}

# 增加图片像素限制，防止处理大图时报错
Image.MAX_IMAGE_PIXELS = None


def create_voc_xml(image_name, width, height, boxes, output_path):
    """创建 PASCAL VOC 格式的 XML 文件"""
    annotation = etree.Element("annotation")

    etree.SubElement(annotation, "folder").text = "images"
    etree.SubElement(annotation, "filename").text = image_name

    source = etree.SubElement(annotation, "source")
    etree.SubElement(source, "database").text = "xView"

    size = etree.SubElement(annotation, "size")
    etree.SubElement(size, "width").text = str(width)
    etree.SubElement(size, "height").text = str(height)
    etree.SubElement(size, "depth").text = "3"

    etree.SubElement(annotation, "segmented").text = "0"

    for box in boxes:
        cls_id, xmin, ymin, xmax, ymax = box

        # 过滤无效框
        if xmin >= xmax or ymin >= ymax:
            continue

        obj = etree.SubElement(annotation, "object")
        # 获取类别名称，如果不在字典里则用 type_id
        class_name = XVIEW_CLASSES.get(cls_id, str(cls_id))
        class_name = class_name.replace(" ", "_").replace("/", "_")  # 规范化名称

        etree.SubElement(obj, "name").text = class_name
        etree.SubElement(obj, "pose").text = "Unspecified"
        etree.SubElement(obj, "truncated").text = "0"
        etree.SubElement(obj, "difficult").text = "0"

        bndbox = etree.SubElement(obj, "bndbox")
        etree.SubElement(bndbox, "xmin").text = str(int(xmin))
        etree.SubElement(bndbox, "ymin").text = str(int(ymin))
        etree.SubElement(bndbox, "xmax").text = str(int(xmax))
        etree.SubElement(bndbox, "ymax").text = str(int(ymax))

    tree = etree.ElementTree(annotation)
    tree.write(output_path, pretty_print=True, xml_declaration=False, encoding="utf-8")


def get_boxes_for_chip(chip_box, all_boxes, min_visibility):
    """
    计算切片内的标注框
    chip_box: [c_xmin, c_ymin, c_xmax, c_ymax] 切片在原图的坐标
    all_boxes: list of [cls_id, x1, y1, x2, y2]
    """
    c_xmin, c_ymin, c_xmax, c_ymax = chip_box
    valid_boxes = []

    for box in all_boxes:
        cls_id, b_xmin, b_ymin, b_xmax, b_ymax = box

        # 计算交集
        inter_xmin = max(c_xmin, b_xmin)
        inter_ymin = max(c_ymin, b_ymin)
        inter_xmax = min(c_xmax, b_xmax)
        inter_ymax = min(c_ymax, b_ymax)

        if inter_xmax > inter_xmin and inter_ymax > inter_ymin:
            # 计算原始面积和交集面积
            box_area = (b_xmax - b_xmin) * (b_ymax - b_ymin)
            inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)

            # 检查可见性 (如果框大部分在切片外，则丢弃)
            if (inter_area / box_area) >= min_visibility:
                # 转换坐标为相对切片的坐标
                new_xmin = inter_xmin - c_xmin
                new_ymin = inter_ymin - c_ymin
                new_xmax = inter_xmax - c_xmin
                new_ymax = inter_ymax - c_ymin

                # 边界检查
                new_xmin = max(0, new_xmin)
                new_ymin = max(0, new_ymin)
                new_xmax = min(c_xmax - c_xmin, new_xmax)
                new_ymax = min(c_ymax - c_ymin, new_ymax)

                valid_boxes.append([cls_id, new_xmin, new_ymin, new_xmax, new_ymax])

    return valid_boxes


def main():
    # 1. 创建目录结构
    jpg_dir = os.path.join(CONFIG["output_dir"], "JPEGImages")
    xml_dir = os.path.join(CONFIG["output_dir"], "Annotations")
    os.makedirs(jpg_dir, exist_ok=True)
    os.makedirs(xml_dir, exist_ok=True)

    print("正在加载 GeoJSON (这可能需要几秒钟)...")
    with open(CONFIG["geojson_path"], 'r') as f:
        data = json.load(f)

    # 2. 整理标注数据: {image_id: [[class, x1, y1, x2, y2], ...]}
    coords_dict = {}
    for feature in tqdm(data['features'], desc="解析标注"):
        props = feature['properties']
        img_id = props['image_id']
        type_id = int(props['type_id'])
        # xView geojson 中的 bounds_imcoords 是像素坐标: xmin, ymin, xmax, ymax
        bounds = [float(x) for x in props['bounds_imcoords'].split(",")]

        if img_id not in coords_dict:
            coords_dict[img_id] = []
        coords_dict[img_id].append([type_id] + bounds)

    # 3. 处理图片
    print(f"开始处理图片并进行切片 (大小: {CONFIG['crop_size']}x{CONFIG['crop_size']})...")

    # 获取目录下所有tif文件
    all_images = [f for f in os.listdir(CONFIG["images_dir"]) if f.endswith('.tif')]

    for img_file in tqdm(all_images, desc="转换进度"):
        if img_file not in coords_dict:
            continue  # 如果该图片没有标注，跳过

        img_path = os.path.join(CONFIG["images_dir"], img_file)
        try:
            # 打开原图
            img = Image.open(img_path)
            w, h = img.size

            step = int(CONFIG["crop_size"] * (1 - CONFIG["overlap"]))

            # 滑动窗口切片
            for y in range(0, h, step):
                for x in range(0, w, step):
                    # 计算切片区域
                    box_w = min(CONFIG["crop_size"], w - x)
                    box_h = min(CONFIG["crop_size"], h - y)

                    # 如果到了边缘，切片大小不足，通常有两种处理：
                    # 1. 放弃 (可能丢失边缘数据) 2. 向回移动起始点 (保持切片尺寸一致)
                    # 这里采用向回移动策略，保证切出来的都是标准大小，除非图本身比crop_size小
                    if box_w < CONFIG["crop_size"] and w > CONFIG["crop_size"]:
                        x = w - CONFIG["crop_size"]
                        box_w = CONFIG["crop_size"]
                    if box_h < CONFIG["crop_size"] and h > CONFIG["crop_size"]:
                        y = h - CONFIG["crop_size"]
                        box_h = CONFIG["crop_size"]

                    chip_rect = (x, y, x + box_w, y + box_h)  # 左上右下

                    # 获取该切片内的标注
                    chip_boxes = get_boxes_for_chip(chip_rect, coords_dict[img_file], CONFIG["min_visibility"])

                    if not chip_boxes and not CONFIG["save_empty"]:
                        continue

                    # 切割图片
                    chip_img = img.crop(chip_rect)

                    # 生成文件名
                    base_name = os.path.splitext(img_file)[0]
                    save_name = f"{base_name}_{x}_{y}"
                    jpg_path = os.path.join(jpg_dir, save_name + ".jpg")
                    xml_path = os.path.join(xml_dir, save_name + ".xml")

                    # 保存 JPG (使用高质量)
                    chip_img.save(jpg_path, quality=95)

                    # 保存 XML
                    create_voc_xml(save_name + ".jpg", box_w, box_h, chip_boxes, xml_path)

        except Exception as e:
            print(f"\n处理图片 {img_file} 时出错: {e}")

    print(f"\n转换完成！数据保存在: {CONFIG['output_dir']}")

if __name__ == "__main__":
    main()
