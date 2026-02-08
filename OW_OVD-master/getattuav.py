import json
import os
import time
from openai import OpenAI
from tqdm import tqdm

# ================= 配置区域 =================

# 请替换为您的 API Key (支持 DeepSeek, OpenAI, SiliconFlow 等)
API_KEY = "sk-eeb0b4e780aa4bb8a0447adc0ec0757c"
BASE_URL = "https://api.deepseek.com"  # 或者其他服务商地址
MODEL_NAME = "deepseek-chat"  # 模型名称

OUTPUT_FILE = "uav_attributes1.json"  # 输出文件名

# ================= SOWOD UAV 33 Classes Mapping =================
# 格式: "SOWOD_Class_Name": "Super_Class_Context"
# 必须与之前的 merge_sowod_voc.py 中的类别名称完全一致 (小写, 连字符)

CLASS_SUPERCLASS_MAP = {
    # --- Task 1: Base (Common Transport & Beings) ---
    "airplane": "Aircraft",
    "ship": "Watercraft",
    "car": "Land Vehicle",
    "truck": "Land Vehicle",
    "bus": "Land Vehicle",
    "van": "Land Vehicle",
    "bridge": "Infrastructure",
    "harbor": "Infrastructure Complex",
    "storage-tank": "Industrial Structure",
    "chimney": "Industrial Structure",
    "pedestrian": "Human",
    "people": "Group of Humans",
    "bicycle": "Small Vehicle",
    "motor": "Small Vehicle",  # Motorcycle

    # --- Task 2: Sports & Leisure ---
    "baseball-field": "Sports Facility",
    "basketball-court": "Sports Facility",
    "tennis-court": "Sports Facility",
    "ground-track-field": "Sports Facility",
    "soccer-ball-field": "Sports Facility",
    "swimming-pool": "Sports Facility",
    "stadium": "Large Sports Facility",

    # --- Task 3: Infrastructure ---
    "airport": "Large Infrastructure Complex",
    "train-station": "Transport Infrastructure",
    "dam": "Water Infrastructure",
    "overpass": "Road Infrastructure",
    "toll-station": "Road Infrastructure",
    "service-area": "Road Infrastructure",
    "roundabout": "Road Infrastructure",

    # --- Task 4: Rare & Difficult ---
    "helicopter": "Aircraft",
    "windmill": "Industrial Structure",
    "golf-field": "Sports Facility",
    "tricycle": "Small Vehicle",
    "awning-tricycle": "Small Vehicle"
}

# =======================================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


# def generate_prompt(class_name, super_class):
#     """
#     [UAV-OWD Specialization: Aerial & Drone Perspective]
#     专为 DIOR/DOTA (航拍) 和 VisDrone (无人机低空) 优化。
#     核心策略：Top-Down View (俯视) + Oblique View (斜视) + Context
#     """
#     clean_name = class_name.replace("-", " ")
#
#     return f"""
# You are an expert Aerial Imagery Analyst specialized in UAV (Unmanned Aerial Vehicle) and Satellite imagery.
# Target Object: "{clean_name}"
# Super-Class Context: This object is a subtype of "{super_class}".
#
# ### MISSION
# Generate exactly **30** distinct visual attributes visible from an **Aerial or Drone View**.
# The images vary from high-altitude satellite (2D shapes) to low-altitude drone (3D structures with side profiles).
#
# ### 🧠 CORE STRATEGY: "Aerial Objectness via 7 Dimensions"
# Scan through these dimensions focused on overhead/oblique imagery.
#
# 1. **2D Footprint/Outline** (The geometric shape seen from above, e.g., circular tank, rectangular court).
# 2. **Roof/Top Structure** (The primary visible surface, e.g., windshield, roof vents, deck).
# 3. **Shadow Projection** (Crucial for height estimation in aerial views).
# 4. **Relative Scale** (Size comparison to roads, cars, or markings).
# 5. **Color/Contrast** (Distinct colors like red tracks, blue pools, white planes).
# 6. **Spatial Context** (Where is it found? e.g., on a runway, in a stadium, on a road).
# 7. **Pattern/Layout** (e.g., lane markings on a track, containers stacked in rows).
#
# ### 💡 FEW-SHOT EXAMPLES (LEARN THE "AERIAL" STYLE)
#
# **Case 1: Vehicles (e.g., Car/Truck)**
# ✅ GOOD (Top/Oblique): "object which has a rectangular roof and a visible windshield"
# ✅ GOOD (Context): "object which is aligned parallel to lane markings on an asphalt road"
# ❌ BAD (Micro-detail): "object which has a license plate number" (Too small to see)
#
# **Case 2: Sports Fields (e.g., Tennis Court)**
# ✅ GOOD (Geometry): "object which consists of a rectangular court divided by white lines"
# ✅ GOOD (Color): "object which typically features a green or blue synthetic surface"
# ❌ BAD (Function): "object which is used to play tennis" (Abstract)
#
# **Case 3: Infrastructure (e.g., Bridge)**
# ✅ GOOD (Structure): "object which is a long linear structure connecting two land masses over water"
# ✅ GOOD (Shadow): "object which casts a long shadow on the water surface below"
#
# ### ⛔ STRICT NEGATIVE CONSTRAINTS
# 1. **NO Invisible Internal Functions:** Don't say "has an engine inside".
# 2. **NO Human Actions:** Don't say "people are driving it".
# 3. **NO Abstract Traits:** No price, speed, or brand names.
#
# ### OUTPUT FORMAT
# Return ONLY a valid JSON object.
# Every line MUST start with "object which ".
#
# {{
#   "{class_name}": [
#     "object which [Footprint: specific top-down shape...]",
#     "object which [Shadow: shadow shape detail...]",
#     "object which [Roof: distinct top detail...]",
#     "object which [Context: specific location...]",
#     ...
#     (Generate exactly 30 lines)
#   ]
# }}
# """

def generate_prompt(class_name, super_class):
    return f"""
You are an expert Computer Vision Data Generator for an Open-World Object Detection model (OW-OVD), specialized in **UAV (Unmanned Aerial Vehicle) and Aerial Imagery**.

Target Object: "{class_name}"
Super-Class Context: This object is a subtype of "{super_class}".

======================================================================
MISSION
Generate exactly **30** distinct visual attributes from an **AERIAL PERSPECTIVE**.
(I will strictly select the best 25.)

Every line MUST start with:
    object which ...

Return ONLY a valid JSON object.
======================================================================

🧠 CORE STRATEGY — NADIR & OBLIQUE VIEW DISCRIMINATION

Each attribute must be identifiable from a **top-down (nadir)** or **high-angle oblique** view.
1. **Topological Objectness:** Focus on the footprint, roof-line, and projected geometry.
2. **Aerial Discrimination:** Distinguish "{class_name}" from other "{super_class}" by features visible from above.

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

            # 兼容性处理：LLM 可能返回 "storage tank" 而不是 "storage-tank"
            clean_name = class_name.replace("-", " ")

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

                # 1. 过滤: 必须以 "object which" 开头
                valid_attrs = [str(a).strip() for a in attrs if str(a).strip().startswith("object which")]

                # 2. 去重
                seen = set()
                unique_attrs = []
                for attr in valid_attrs:
                    if attr not in seen:
                        unique_attrs.append(attr)
                        seen.add(attr)

                # 3. 数量检查
                if len(unique_attrs) < 20:
                    print(f"  ⚠️ Warning: Only {len(unique_attrs)} attributes for {class_name}. Retrying...")
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
            with open(OUTPUT_FILE, 'r') as f:
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
            all_data[cls_name] = attrs

            # 实时保存
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(all_data, f, indent=2)

            # 避免触发 API 速率限制
            time.sleep(0.5)
        else:
            print(f"\nFailed to generate attributes for {cls_name}")

    print(f"\n✅ All Done! Data saved to {OUTPUT_FILE}")
    print(f"Next Step: Update 'ATTRIBUTES_FILE' path in 'generate_sowod_uva_uav.py' to point to this file.")


if __name__ == "__main__":
    main()