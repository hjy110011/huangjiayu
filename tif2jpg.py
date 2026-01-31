from PIL import Image
import os

src_dir = "D:\\xview\\images\\train_images"
dst_dir = "D:\\xview\\VOC\\JPEGImages\\"
os.makedirs(dst_dir, exist_ok=True)

for img_name in os.listdir(src_dir):
    if img_name.endswith(".tif"):
        img = Image.open(os.path.join(src_dir, img_name))
        img = img.convert("RGB")
        img.save(os.path.join(dst_dir, img_name.replace(".tif", ".jpg")))
