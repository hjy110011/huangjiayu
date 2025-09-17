# yaml_to_graph_hier.py

import yaml
import networkx as nx
import matplotlib.pyplot as plt

def draw_yolo_yaml(yaml_path, out_png="yolo_yaml_arch.png"):
    # 读取 yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 合并 backbone 和 head
    layers_all = []
    for part in ["backbone", "head"]:
        if part in cfg:
            for idx, layer in enumerate(cfg[part]):
                layers_all.append((part, idx, layer))

    G = nx.DiGraph()

    # 添加节点和边
    for global_idx, (part, local_idx, layer) in enumerate(layers_all):
        f, n, m, args = layer
        node_name = f"{global_idx}_{m}"
        G.add_node(node_name, label=f"{global_idx}: {m}\n×{n}\n{args}", layer=global_idx)

        # from 字段
        if isinstance(f, int):
            sources = [f]
        else:
            sources = f

        for s in sources:
            if s == -1:  # 上一层
                src_idx = global_idx - 1
                if src_idx >= 0:
                    src_name = f"{src_idx}_{layers_all[src_idx][2][2]}"
                    G.add_edge(src_name, node_name)
            elif 0 <= s < len(layers_all):
                src_name = f"{s}_{layers_all[s][2][2]}"
                G.add_edge(src_name, node_name)
            else:
                print(f"⚠️ 警告: 层 {global_idx} 的来源索引 {s} 超出范围")

    # 手动布局：纵向位置 = layer index，横向按模块分类
    pos = {}
    x_offsets = {}
    for node, data in G.nodes(data=True):
        layer = data["layer"]
        module = node.split("_", 1)[1]  # 模块名
        if module not in x_offsets:
            x_offsets[module] = len(x_offsets)  # 每种模块分配一个横向位置
        pos[node] = (x_offsets[module], -layer)  # y 用负号保证从上到下

    # 绘图
    plt.figure(figsize=(14, 12))
    nx.draw(G, pos,
            with_labels=True,
            labels=nx.get_node_attributes(G, "label"),
            node_size=2800,
            node_color="#87CEFA",
            font_size=7,
            font_weight="bold",
            arrows=True)

    plt.title(f"YOLO Network Structure from {yaml_path}", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.show()
    print(f"结构图已保存为 {out_png}")



if __name__ == "__main__":
    # 修改为你本地的 yaml，例如 ultralytics/cfg/models/v8/yolov8s.yaml
    yaml_file = "/home/gdut-627/huangjiayu/ultralytics-main/ultralytics/cfg/models/MyModels/MyYolo1.yaml"
    draw_yolo_yaml(yaml_file)
