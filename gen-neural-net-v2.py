#######################################################################################################
### prompt for chatGPT6.0: 帮我生成一段用python画出输入层为10个节点、中间层为8个节点、               ###
###                        输出层为10个节点的神经网络图的代码，节点自左向右展示，用细线相连接。     ###
#######################################################################################################
import matplotlib.pyplot as plt

# 网络结构
layers = [10, 8, 10]
layer_names = ['Input Layer', 'Hidden Layer', 'Output Layer']

# 各层横坐标
x_positions = [0, 2.5, 5]

# 保存每层节点坐标
node_positions = []

# 计算每层节点位置
for layer_idx, node_count in enumerate(layers):
    x = x_positions[layer_idx]

    # 让每层节点在垂直方向居中
    y_positions = [
        i - (node_count - 1) / 2
        for i in range(node_count)
    ]

    node_positions.append([
        (x, y) for y in y_positions
    ])

# 创建画布
fig, ax = plt.subplots(figsize=(12, 8))

# -------------------------
# 1. 绘制连接线
# -------------------------
for layer_idx in range(len(layers) - 1):

    current_layer = node_positions[layer_idx]
    next_layer = node_positions[layer_idx + 1]

    for x1, y1 in current_layer:
        for x2, y2 in next_layer:
            ax.plot(
                [x1, x2],
                [y1, y2],
                linewidth=0.35,   # 细线
                alpha=0.45
            )

# -------------------------
# 2. 绘制节点
# -------------------------
for layer_idx, layer in enumerate(node_positions):
    for node_idx, (x, y) in enumerate(layer):

        circle = plt.Circle(
            (x, y),
            radius=0.16,
            fill=True,
            edgecolor='black',
            linewidth=1,
            zorder=3
        )

        ax.add_patch(circle)

# -------------------------
# 3. 添加层名称
# -------------------------
max_nodes = max(layers)

for i, name in enumerate(layer_names):
    ax.text(
        x_positions[i],
        max_nodes / 2 + 0.7,
        name,
        ha='center',
        va='center',
        fontsize=13
    )

# -------------------------
# 4. 添加节点数量说明
# -------------------------
for i, node_count in enumerate(layers):
    ax.text(
        x_positions[i],
        -max_nodes / 2 - 0.8,
        f'{node_count} nodes',
        ha='center',
        fontsize=11
    )

# -------------------------
# 5. 调整显示效果
# -------------------------
ax.set_xlim(-0.8, 5.8)
ax.set_ylim(-6, 6)

ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.show()
