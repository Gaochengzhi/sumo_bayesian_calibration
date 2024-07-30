import numpy as np
import matplotlib.pyplot as plt

# 假设你的 data 已经存在
data = np.random.randn(1000)  # 示例数据

# 使用 np.histogram 计算直方图
hist_data, bins = np.histogram(data, bins=100, density=True)

# 计算每个 bin 的宽度
bin_width = bins[1] - bins[0]

# 计算每个 bin 的中心点
bin_centers = (bins[:-1] + bins[1:]) / 2

# 使用 plt.bar 绘制直方图
plt.figure(figsize=(10, 6))
plt.bar(bin_centers, hist_data, width=bin_width, align="center", edgecolor="k")

# 添加一些图表样式
plt.xlabel("Bin Centers")
plt.ylabel("Density")
plt.title("Histogram using plt.bar")
plt.savefig("dsa.png")
