import os
import json
import shutil
from PIL import Image
from lxml import etree
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, freeze_support

# ================= 配置区域 =================
CONFIG = {
    "geojson_path": "D:\\xview\\xView_train.geojson",
    "images_dir": "D:\\xview\\images\\train_images\\",
    "output_dir": "D:\\xview\\xView_VOC1",
    "crop_size": 800,
    "overlap": 0.2,
    "min_visibility": 0.7,  # 建议稍微降低一点，0.8可能太严格导致丢框
    "save_empty": False,
    "num_workers": min(8, cpu_count())  # 并行进程数，防止卡死
}

Image.MAX_IMAGE_PIXELS = None

# xView 类别映射 (保持不变)
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


def create_voc_xml(image_name, width, height, boxes, output_path):
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
        # 边界保护
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(width, xmax), min(height, ymax)

        if xmin >= xmax or ymin >= ymax: continue

        obj = etree.SubElement(annotation, "object")
        class_name = XVIEW_CLASSES.get(cls_id, str(cls_id)).replace(" ", "_").replace("/", "_")

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
    c_xmin, c_ymin, c_xmax, c_ymax = chip_box
    valid_boxes = []

    for box in all_boxes:
        cls_id, b_xmin, b_ymin, b_xmax, b_ymax = box

        inter_xmin = max(c_xmin, b_xmin)
        inter_ymin = max(c_ymin, b_ymin)
        inter_xmax = min(c_xmax, b_xmax)
        inter_ymax = min(c_ymax, b_ymax)

        if inter_xmax > inter_xmin and inter_ymax > inter_ymin:
            box_area = (b_xmax - b_xmin) * (b_ymax - b_ymin)
            inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)

            # 保护除零错误
            if box_area <= 0: continue

            if (inter_area / box_area) >= min_visibility:
                new_xmin = inter_xmin - c_xmin
                new_ymin = inter_ymin - c_ymin
                new_xmax = inter_xmax - c_xmin
                new_ymax = inter_ymax - c_ymin
                valid_boxes.append([cls_id, new_xmin, new_ymin, new_xmax, new_ymax])
    return valid_boxes


# 将处理单张图片的逻辑封装，以便多进程调用
def process_single_image(args):
    img_file, boxes_info, config = args
    img_path = os.path.join(config["images_dir"], img_file)

    jpg_dir = os.path.join(config["output_dir"], "JPEGImages")
    xml_dir = os.path.join(config["output_dir"], "Annotations")

    try:
        img = Image.open(img_path)
        w, h = img.size
        step = int(config["crop_size"] * (1 - config["overlap"]))

        processed_count = 0

        # 这里的循环逻辑与原代码一致
        for y in range(0, h, step):
            for x in range(0, w, step):
                box_w = min(config["crop_size"], w - x)
                box_h = min(config["crop_size"], h - y)

                # 边缘回退逻辑
                if box_w < config["crop_size"] and w > config["crop_size"]:
                    x = w - config["crop_size"]
                    box_w = config["crop_size"]
                if box_h < config["crop_size"] and h > config["crop_size"]:
                    y = h - config["crop_size"]
                    box_h = config["crop_size"]

                chip_rect = (x, y, x + box_w, y + box_h)
                chip_boxes = get_boxes_for_chip(chip_rect, boxes_info, config["min_visibility"])

                if not chip_boxes and not config["save_empty"]:
                    continue

                # === 关键修复：颜色模式 ===
                chip_img = img.crop(chip_rect)
                if chip_img.mode != "RGB":
                    chip_img = chip_img.convert("RGB")

                base_name = os.path.splitext(img_file)[0]
                save_name = f"{base_name}_{x}_{y}"

                chip_img.save(os.path.join(jpg_dir, save_name + ".jpg"), quality=90)
                create_voc_xml(save_name + ".jpg", box_w, box_h, chip_boxes, os.path.join(xml_dir, save_name + ".xml"))
                processed_count += 1

        return f"Done: {img_file} ({processed_count} chips)"
    except Exception as e:
        return f"Error {img_file}: {e}"


def main():
    freeze_support()  # Windows下多进程必需

    # 目录准备
    jpg_dir = os.path.join(CONFIG["output_dir"], "JPEGImages")
    xml_dir = os.path.join(CONFIG["output_dir"], "Annotations")
    os.makedirs(jpg_dir, exist_ok=True)
    os.makedirs(xml_dir, exist_ok=True)

    print("Loading GeoJSON...")
    with open(CONFIG["geojson_path"], 'r') as f:
        data = json.load(f)

    print("Parsing annotations...")
    coords_dict = {}
    for feature in tqdm(data['features']):
        props = feature['properties']
        img_id = props.get('image_id')  # 使用 .get 防止报错
        if not img_id: continue

        type_id = int(props['type_id'])
        bounds = [float(x) for x in props['bounds_imcoords'].split(",")]

        if img_id not in coords_dict:
            coords_dict[img_id] = []
        coords_dict[img_id].append([type_id] + bounds)

    all_images = [f for f in os.listdir(CONFIG["images_dir"]) if f.endswith('.tif')]
    # 过滤掉没有标注的图片，减少计算
    target_images = [f for f in all_images if f in coords_dict]

    print(f"Start processing {len(target_images)} images with {CONFIG['num_workers']} workers...")

    # 准备多进程参数
    process_args = [(f, coords_dict[f], CONFIG) for f in target_images]

    with Pool(CONFIG["num_workers"]) as p:
        # 使用 imap_unordered 可以配合 tqdm 显示进度
        list(tqdm(p.imap_unordered(process_single_image, process_args), total=len(target_images)))

    print("All Done!")


if __name__ == "__main__":
    main()