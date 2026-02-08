import json
import os
import time
from openai import OpenAI
from tqdm import tqdm

# ================= Configuration Area =================

# Recommended: DeepSeek (Strong logic & cheap) or SiliconFlow/Qwen
API_KEY = "sk-eeb0b4e780aa4bb8a0447adc0ec0757c"  # Your API Key
BASE_URL = "https://api.deepseek.com"  # DeepSeek Base URL
MODEL_NAME = "deepseek-chat"  # Model Name

OUTPUT_FILE = "xview_attributes2.json"

# ================= xView 60 Classes Mapping (15-15-15-15 Split) =================
# Format: "Class_Name_In_Dataset": "Super_Class_Context"
CLASS_SUPERCLASS_MAP = {
    # --- Task 1: General Mobility (15 classes) ---
    "Fixed-wing_Aircraft": "Aircraft",
    "Small_Aircraft": "Aircraft",
    "Cargo_Plane": "Aircraft",
    "Helicopter": "Aircraft",
    "Passenger_Vehicle": "Land Vehicle",
    "Small_Car": "Land Vehicle",
    "Bus": "Land Vehicle",
    "Pickup_Truck": "Land Vehicle",
    "Truck": "Land Vehicle",  # Generic truck
    "Railway_Vehicle": "Train Car",
    "Passenger_Car": "Train Car",  # Rail passenger car
    "Locomotive": "Train Engine",
    "Maritime_Vessel": "Watercraft",
    "Motorboat": "Watercraft",
    "Fishing_Vessel": "Watercraft",

    # --- Task 2: Logistics (15 classes) ---
    "Utility_Truck": "Industrial Vehicle",
    "Cargo_Truck": "Industrial Vehicle",
    "Truck_w_Box": "Industrial Vehicle",
    "Truck_Tractor": "Industrial Vehicle",  # Semi-truck head
    "Trailer": "Vehicle Attachment",
    "Truck_w_Flatbed": "Industrial Vehicle",
    "Truck_w_Liquid": "Industrial Vehicle",
    "Crane_Truck": "Construction Vehicle",
    "Ferry": "Watercraft",
    "Yacht": "Watercraft",
    "Container_Ship": "Large Vessel",
    "Oil_Tanker": "Large Vessel",
    "Engineering_Vehicle": "Construction Vehicle",
    "Dump_Truck": "Construction Vehicle",
    "Haul_Truck": "Industrial Vehicle",

    # --- Task 3: Engineering (15 classes) ---
    "Tower_crane": "Construction Equipment",
    "Container_Crane": "Port Equipment",
    "Reach_Stacker": "Port Vehicle",
    "Straddle_Carrier": "Port Vehicle",
    "Mobile_Crane": "Construction Vehicle",
    "Scraper_Tractor": "Construction Vehicle",
    "Front_loader_Bulldozer": "Construction Vehicle",
    "Excavator": "Construction Equipment",
    "Cement_Mixer": "Construction Vehicle",
    "Ground_Grader": "Construction Vehicle",
    "Hut_Tent": "Temporary Structure",
    "Shed": "Small Structure",
    "Construction_Site": "Land Use Area",
    "Vehicle_Lot": "Land Use Area",
    "Helipad": "Infrastructure",

    # --- Task 4: Static & Infra (15 classes) ---
    "Building": "Structure",
    "Aircraft_Hangar": "Large Structure",
    "Damaged_Building": "Structure",
    "Facility": "Large Infrastructure Complex",
    "Storage_Tank": "Industrial Structure",
    "Shipping_container_lot": "Logistics Area",
    "Shipping_Container": "Logistics Object",
    "Pylon": "Infrastructure Pole",
    "Tower": "Vertical Structure",
    "Flat_Car": "Train Car",
    "Tank_car": "Train Car",
    "Cargo_Car": "Train Car",
    "Sailboat": "Watercraft",
    "Tugboat": "Watercraft",
    "Barge": "Watercraft"
}

# =======================================================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

#
# def generate_prompt(class_name, super_class):
#     """
#     [V19 - Schema-Aligned + Few-Shot Guidance]
#     包含：
#     1. Discriminative Objectness 策略。
#     2. 9大视觉维度锚点。
#     3. 严格的正负样本对比 (Few-Shot)，纠正抽象描述。
#     """
#     return f"""
# You are an expert Computer Vision Data Generator for an Open-World Object Detection model (OW-OVD).
# Target Object: "{class_name}"
# Super-Class Context: This object is a subtype of "{super_class}".
#
# ### MISSION
# Generate exactly **30** distinct visual attributes.
# (I will strictly select the best 25, so provide high-quality, diverse candidates).
#
# ### 🧠 CORE STRATEGY: "Discriminative Objectness via 9 Dimensions"
# Scan through these 9 dimensions. For each, describe visible physical traits (Objectness) and unique details (Discrimination).
# **Do NOT suppress generic parent traits** (Inheritance is allowed).
#
# 1. **Shape** (Geometric forms, outlines)
# 2. **Color** (Dominant colors, patterns)
# 3. **Texture** (Surface quality, tactility)
# 4. **Size** (Relative scale, aspect ratio)
# 5. **Material** (Physical composition)
# 6. **Features** (Distinct visible parts - *Zoom in here*)
# 7. **Appearance** (Overall look/style)
# 8. **Behavior** (Visible states/poses - *Must be visual*)
# 9. **Context/Environment** (Typical background)
#
# ### 💡 FEW-SHOT EXAMPLES (LEARN FROM THESE)
# Study these pairs to understand the required "Visual Style":
#
# **Case 1: Handling "Function/Behavior" (Crucial)**
# ❌ BAD (Abstract Function): "object which is used for drinking coffee"
# ✅ GOOD (Visualized State): "object which has a hollow cylindrical body to hold liquid"
# ✅ GOOD (Visualized Interaction): "object which is held by a human hand via a handle"
#
# **Case 2: Handling "Context"**
# ❌ BAD (Invisible Context): "object which is sold in a supermarket"
# ✅ GOOD (Visual Background): "object which is placed on a shelf or table surface"
#
# **Case 3: Handling "Subjectivity"**
# ❌ BAD (Subjective): "object which looks beautiful and expensive"
# ✅ GOOD (Objective): "object which has a glossy, polished gold surface"
#
# **Case 4: Handling "Generic Traits"**
# ❌ BAD (Too Generic): "object which is an object"
# ✅ GOOD (Generic but Visual): "object which casts a shadow on the ground"
#
# ### ⛔ STRICT NEGATIVE CONSTRAINTS
# 1. **NO Abstract Functions:** Never describe what it *does*, describe what it *looks like* when doing it.
# 2. **NO Intangible Traits:** No price, origin (e.g., "made in China"), or sound.
# 3. **NO Redundancy:** Do not generate the same attribute twice with slightly different words.
#
# ### OUTPUT FORMAT
# Return ONLY a valid JSON object.
# Every line MUST start with "object which ".
#
# {{
#   "{class_name}": [
#     "object which [Shape: specific geometric detail...]",
#     "object which [Material: specific material detail...]",
#     "object which [Feature: distinct part detail...]",
#     "object which [Behavior: visible state...]",
#     ...
#     (Generate exactly 30 lines covering the 9 dimensions)
#   ]
# }}
# """

# def generate_prompt(class_name, super_class):
#     """
#     [V20 - xView Specialization: Satellite/Aerial Perspective]
#     专为 xView 数据集优化。
#     核心策略：Top-Down View (BEV) + Shadow Reasoning + Contextual Scale
#     """
#     return f"""
# You are an expert Aerial Imagery Analyst specialized in the xView dataset (Satellite/Overhead Imagery).
# Target Object: "{class_name}"
# Super-Class Context: This object is a subtype of "{super_class}".
#
# ### MISSION
# Generate exactly **30** distinct visual attributes strictly from a **Top-Down / Bird's Eye View (BEV)**.
# The images are satellite photos, so objects often look like 2D geometric shapes with shadows.
#
# ### 🧠 CORE STRATEGY: "Aerial Objectness via 7 Dimensions"
# Scan through these 7 dimensions focused on overhead imagery.
# **CRITICAL:** Do NOT describe features only visible from the ground (e.g., tires, side doors, license plates, faces).
#
# 1. **2D Footprint/Outline** (The geometric shape seen from above, e.g., rectangular, T-shaped).
# 2. **Roof/Top Structure** (The primary visible surface, e.g., vents on a roof, sunroof on a car).
# 3. **Shadow Projection** (The shadow cast on the ground, indicating height and side profile).
# 4. **Relative Scale** (Size comparison to roads, buildings, or other vehicles).
# 5. **Color/Reflectivity** (Sun glint, matte paint, camouflage patterns).
# 6. **Spatial Context** (Where is it usually found? e.g., on asphalt, moored at a dock, in a construction zone).
# 7. **Orientation/Alignment** (e.g., aligned parallel to road lanes, parked in rows).
#
# ### 💡 FEW-SHOT EXAMPLES (LEARN THE "AERIAL" STYLE)
# Study these pairs to strictly separate "Ground View" (Invalid) from "Aerial View" (Valid):
#
# **Case 1: Handling "Vehicles" (e.g., Truck)**
# ❌ BAD (Ground View): "object which has large circular tires and a front grille"
# ✅ GOOD (Top-Down): "object which consists of a distinct cabin block followed by a long rectangular trailer"
# ✅ GOOD (Shadow): "object which casts a rectangular shadow indicating a boxy vertical profile"
#
# **Case 2: Handling "Maritime" (e.g., Ship)**
# ❌ BAD (Waterline View): "object which has a hull rising out of the water"
# ✅ GOOD (Top-Down): "object which has a pointed bow and a flat deck surface contrasting with the dark water"
# ✅ GOOD (Wake): "object which leaves a V-shaped white wake trail on the water surface"
#
# **Case 3: Handling "Buildings/Infrastructure"**
# ❌ BAD (Facade View): "object which has glass windows and a front entrance"
# ✅ GOOD (Top-Down): "object which has a flat grey concrete roof with AC units visible on top"
# ✅ GOOD (Outline): "object which forms a large L-shaped or U-shaped geometric footprint"
#
# **Case 4: Handling "Generic/Low-Res Features"**
# ❌ BAD (Too Detailed): "object which has a brand logo on the hood"
# ✅ GOOD (Resolution Aware): "object which appears as a small, compact rectangular blob on the road"
#
# ### ⛔ STRICT NEGATIVE CONSTRAINTS
# 1. **NO Ground-Level Features:** Never mention wheels, undercarriages, side windows, doors, or license plates.
# 2. **NO Human Actions:** Do not say "held by a person" (people are dots). Say "located near small pixelated clusters (people)".
# 3. **NO Intangible Traits:** No functions ("used for transport"), only visual evidence ("located on a highway").
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
#     (Generate exactly 30 lines covering the Aerial dimensions)
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
                     "content": "You are a specialized data generation assistant for computer vision. You output strictly valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"},
                max_tokens=1024
            )

            content = response.choices[0].message.content

            # Clean up markdown
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")
            elif content.startswith("```"):
                content = content.replace("```", "")

            content = content.strip()

            # Parse JSON
            data = json.loads(content)

            # Validate structure
            # The prompt asks for JSON with key {class_name} (with underscores)
            # But sometimes LLM might output the cleaned name. We check both.
            clean_name = class_name.replace("_", " ")

            target_key = None
            if class_name in data:
                target_key = class_name
            elif clean_name in data:
                target_key = clean_name

            if target_key and isinstance(data[target_key], list):
                attrs = data[target_key]

                # 1. Filter: Must start with "object which"
                valid_attrs = [a for a in attrs if str(a).strip().startswith("object which")]

                # 2. Deduplicate
                seen = set()
                unique_attrs = []
                for attr in valid_attrs:
                    if attr not in seen:
                        unique_attrs.append(attr)
                        seen.add(attr)

                # 3. Check count
                if len(unique_attrs) < 20:
                    print(f"  Warning: Only got {len(unique_attrs)} valid attributes for {class_name}")

                # Take top 25
                final_attrs = unique_attrs[:25]

                return final_attrs
            else:
                print(
                    f"  Warning: JSON structure invalid for {class_name}, keys found: {list(data.keys())}. Retrying...")

        except json.JSONDecodeError:
            print(f"  Error: Failed to parse JSON for {class_name}. Retrying ({attempt + 1}/{max_retries})...")
        except Exception as e:
            print(f"  Error fetching {class_name}: {e}. Retrying ({attempt + 1}/{max_retries})...")
            time.sleep(2)

    return None


def main():
    print(f"Target Output File: {OUTPUT_FILE}")

    # 1. Resume logic
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

    # 2. Processing Loop
    pbar = tqdm(CLASS_SUPERCLASS_MAP.items())
    for cls_name, super_cls in pbar:
        # Check if already done (using the dataset class name with underscores)
        if cls_name in all_data and len(all_data[cls_name]) >= 20:
            continue

        pbar.set_description(f"Generating: {cls_name}")

        attrs = fetch_attributes(cls_name, super_cls)

        if attrs:
            all_data[cls_name] = attrs  # Save using the underscore name

            # Save continuously
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(all_data, f, indent=2)

            time.sleep(0.5)  # Rate limit protection
        else:
            print(f"\nFailed to generate attributes for {cls_name} after retries.")

    print(f"\nAll Done! Data saved to {OUTPUT_FILE}")
    print(f"Next Step: Rename {OUTPUT_FILE} to 'attributes.json' (or update config) and run your embedding script.")


if __name__ == "__main__":
    main()