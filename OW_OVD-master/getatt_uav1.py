import json
import os
import time
import re
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================

# 请替换为您的 API Key (支持 DeepSeek, OpenAI, SiliconFlow 等)
API_KEY = "sk-0391a10502ec4c0ab4ab9ac1eda3905b"
BASE_URL = "https://api.deepseek.com"  # 或者其他服务商地址
MODEL_NAME = "deepseek-chat"  # 模型名称

OUTPUT_FILE = "uav_attributes3.json"  # 输出文件名

# ================= SOWOD UAV 33 Classes Mapping (10-8-7-8) =================
# 格式: "SOWOD_Class_Name": "Super_Class_Context"
# 必须与 SOWOD_TASKS 中的类别名称完全一致 (小写, 连字符)

CLASS_SUPERCLASS_MAP = {
    # --- Task 1: Base (10 classes) ---
    "car": "Land Vehicle",
    "truck": "Land Vehicle",
    "bus": "Land Vehicle",
    "van": "Land Vehicle",
    "pedestrian": "Human",
    "people": "Group of Humans",
    "bicycle": "Small Vehicle",
    "motor": "Small Motorcycle",
    "tricycle": "Small Vehicle",
    "awning-tricycle": "Small Vehicle",

    # --- Task 2: AeroWaterInd (8 classes) ---
    "airplane": "Aircraft",
    "helicopter": "Aircraft",
    "ship": "Watercraft",
    "harbor": "Infrastructure Complex",
    "bridge": "Infrastructure",
    "storage-tank": "Industrial Structure",
    "chimney": "Industrial Structure",
    "dam": "Water Infrastructure",

    # --- Task 3: Infra (7 classes) ---
    "airport": "Large Infrastructure Complex",
    "train-station": "Transport Infrastructure",
    "overpass": "Road Infrastructure",
    "toll-station": "Road Infrastructure",
    "service-area": "Road Infrastructure",
    "roundabout": "Road Infrastructure",
    "windmill": "Industrial Structure",

    # --- Task 4: Sports (8 classes) ---
    "baseball-field": "Sports Facility",
    "basketball-court": "Sports Facility",
    "tennis-court": "Sports Facility",
    "ground-track-field": "Sports Facility",
    "soccer-ball-field": "Sports Facility",
    "swimming-pool": "Sports Facility",
    "stadium": "Large Sports Facility",
    "golf-field": "Sports Facility"
}

# =======================================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def generate_prompt(class_name, super_class):
    # 将连字符转为空格，帮助大模型更好地理解语义
    clean_name = class_name.replace("-", " ")

    return f"""
You are an expert Computer Vision Data Generator for an Open-World Object Detection model (OW-OVD), specialized in **UAV (Unmanned Aerial Vehicle) and Aerial Imagery**.

Target Object: "{clean_name}"
Super-Class Context: This object is a subtype of "{super_class}".

======================================================================
MISSION
Generate exactly **30** distinct visual attributes from an **AERIAL PERSPECTIVE**.
(I will strictly select the best 25.)

Every line MUST start with exactly:
    object which ...

Return ONLY a valid JSON object.
======================================================================

🧠 CORE STRATEGY — NADIR & OBLIQUE VIEW DISCRIMINATION

Each attribute must be identifiable from a **top-down (nadir)** or **high-angle oblique** view.
1. **Topological Objectness:** Focus on the footprint, roof-line, and projected geometry.
2. **Aerial Discrimination:** Distinguish "{clean_name}" from other "{super_class}" by features visible from above.

----------------------------------------------------------------------
🧭 9 AERIAL VISUAL DIMENSIONS — USE AS ANCHORS

1. **Top-Down Shape & Footprint**
   (rectilinear outline, circular footprint, cross-shaped planform)

2. **Projected Geometry & Height Evidence** ★ CRITICAL
   (vertical extrusion visible in oblique views, cast shadow indicating height/shape)

3. **Roof/Top Surface Characteristics**
   (top-side texture, skylights, cooling units, solar panels, hatch patterns)

4. **Planar Aspect Ratio**
   (elongated ribbon-like shape vs. compact polygonal form from above)

5. **Structural Symmetry (Aerial)**
   (bilateral symmetry along the longitudinal axis, radial symmetry)

6. **Material Reflectivity & Albedo**
   (specular glint from glass/metal surfaces, matte asphalt-like texture, heat-absorbent dark surfaces)

7. **Boundary Continuity**
   (distinct edge contrast against terrain, paved borders, containment walls)

8. **Orientation & Grouping Patterns**
   (object which aligns in parallel rows, object which forms a cluster with uniform spacing)

9. **Geometric Relation to Ground/Infrastructure**
   (object which is intersected by linear markings, object which fits within a standard parking/docking bay)

----------------------------------------------------------------------
⛔ STRICT UAV-SPECIFIC CONSTRAINTS

• NO eye-level-only details (e.g., "object which has a front-facing door handle").
• NO undercarriage details (unless visible during banking/turning).
• NO small-scale text or labels (usually invisible from flight altitudes).
• NO background-dependent context (people, indoor furniture).
• NO abstract functions or invisible traits.

----------------------------------------------------------------------
OUTPUT FORMAT — JSON ONLY

{{
  "{class_name}": [
    "object which ...",
    ...
    (generate exactly 30 lines)
  ]
}}
"""


def fetch_attributes(class_name, super_class, max_retries=3):
    prompt = generate_prompt(class_name, super_class)
    clean_name = class_name.replace("-", " ")

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system",
                     "content": "You are a specialized data generation assistant. You output strictly valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"},
                max_tokens=1024
            )

            content = response.choices[0].message.content

            # 清理 Markdown 代码块标记
            if content.strip().startswith("```json"):
                content = content.strip().replace("```json", "").replace("```", "")
            elif content.strip().startswith("```"):
                content = content.strip().replace("```", "")

            # 解析 JSON
            data = json.loads(content)

            # 寻找对应的 Key
            target_key = None
            if class_name in data:
                target_key = class_name
            elif clean_name in data:
                target_key = clean_name

            # 如果都没找到，尝试取第一个 key
            if not target_key and data:
                target_key = list(data.keys())[0]

            if target_key and isinstance(data[target_key], list):
                attrs = data[target_key]
                valid_attrs = []

                # 1. 清洗与过滤
                for a in attrs:
                    a_str = str(a).strip()
                    # 强力清洗: 去除行首可能出现的数字序号、点、破折号、星号 (如 "1. ", "- ", "* ")
                    a_str = re.sub(r'^[\d\.\-\*]*\s*', '', a_str)

                    if a_str.lower().startswith("object which"):
                        valid_attrs.append(a_str)

                # 2. 去重
                seen = set()
                unique_attrs = []
                for attr in valid_attrs:
                    if attr not in seen:
                        unique_attrs.append(attr)
                        seen.add(attr)

                # 3. 数量检查
                if len(unique_attrs) < 20:
                    print(f"  ⚠️ Warning: Only {len(unique_attrs)} valid attributes for {class_name}. Retrying...")
                    continue  # 重新尝试生成更多

                # 取前 25 条
                final_attrs = unique_attrs[:25]
                return final_attrs
            else:
                print(f"  ⚠️ Invalid JSON structure for {class_name}. Keys: {list(data.keys())}")

        except json.JSONDecodeError:
            print(f"  ❌ JSON Parse Error for {class_name}. Retrying ({attempt + 1}/{max_retries})...")
        except Exception as e:
            print(f"  ❌ API Error for {class_name}: {e}. Retrying ({attempt + 1}/{max_retries})...")
            time.sleep(2)

    return None


def main():
    print(f"Target Output File: {OUTPUT_FILE}")

    # 1. 断点续传逻辑
    if os.path.exists(OUTPUT_FILE):
        print("Found existing file, loading...")
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                all_data = json.load(f)
            print(f"Loaded {len(all_data)} classes from existing file.")
        except json.JSONDecodeError:
            print("Existing file is corrupted, starting fresh.")
            all_data = {}
    else:
        all_data = {}

    print(f"Starting generation for {len(CLASS_SUPERCLASS_MAP)} classes...")

    # 2. 处理循环
    pbar = tqdm(CLASS_SUPERCLASS_MAP.items())
    for cls_name, super_cls in pbar:
        # 如果已经生成过且数量足够，跳过
        if cls_name in all_data and len(all_data[cls_name]) >= 20:
            continue

        pbar.set_description(f"Generating: {cls_name}")

        attrs = fetch_attributes(cls_name, super_cls)

        if attrs:
            # 强制存为带连字符的标准 SOWOD Class Name
            all_data[cls_name] = attrs

            # 实时保存
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, indent=4)

            # 避免触发 API 速率限制
            time.sleep(0.5)
        else:
            print(f"\nFailed to generate attributes for {cls_name}")

    print(f"\n✅ All Done! Data saved to {OUTPUT_FILE}")
    print(f"Next Step: Use this file in 'generate_sowod_uav.py'.")


if __name__ == "__main__":
    main()