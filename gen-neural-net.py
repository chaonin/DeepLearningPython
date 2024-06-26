#######################################################################################################
### prompt for chatGPT3.5: 帮我生成一段用python画出输入层为7个节点、中间层为5个节点、               ###
###                        输出层为10个节点的神经网络图的代码，节点自左向右展示，用细线相连接。     ###
#######################################################################################################
import matplotlib.pyplot as plt

# 定义神经网络结构
input_layer = 20
hidden_layer = 8 
output_layer = 10

# 计算每层节点之间的垂直间距
vertical_distance = 1.0 / max(input_layer, hidden_layer, output_layer)

# 创建画布和子图
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制输入层节点
for i in range(input_layer):
    ax.scatter(0, 1 - i * vertical_distance, color='blue', s=300)
    ax.text(0, 1 - i * vertical_distance, f'Input {i + 1}', ha='right', va='center')

# 绘制中间层节点
for i in range(hidden_layer):
    ax.scatter(0.5, 1 - i * vertical_distance, color='green', s=300)
    ax.text(0.5, 1 - i * vertical_distance, f'Hidden {i + 1}', ha='right', va='center')

# 绘制输出层节点
for i in range(output_layer):
    ax.scatter(1, 1 - i * vertical_distance, color='red', s=300)
    ax.text(1, 1 - i * vertical_distance, f'Output {i + 1}', ha='left', va='center')

# 绘制连接线
for i in range(input_layer):
    for j in range(hidden_layer):
        ax.plot([0, 0.5], [1 - i * vertical_distance, 1 - j * vertical_distance], color='black', linewidth=0.5)

for i in range(hidden_layer):
    for j in range(output_layer):
        ax.plot([0.5, 1], [1 - i * vertical_distance, 1 - j * vertical_distance], color='black', linewidth=0.5)

# 隐藏坐标轴
ax.axis('off')

plt.show()
