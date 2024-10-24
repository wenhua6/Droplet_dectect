import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

# 存储所有水力学直径的列表
all_diameters = []

# 图片的像素尺寸
image_width = 1280
image_height = 800

# 每个像素的尺寸（微米）
pixel_size = 1.7

# 帧率
Fps = 1000

# 统计区域的深度（微米）
depth = 200

# 遍历文件夹中的所有 txt 文件
folder_path = './runs/detect/exp19/labels'  # 文件夹路径

total_images = 0  # 总图片数量

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)

        # 读取文件并解析矩形信息
        rectangles = []
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 5:
                    rectangles.append([float(part) for part in parts[1:]])

        # 计算液滴直径（微米）
        diameters = [(rect[2] * image_width + rect[3] * image_height) / 2 * pixel_size for rect in rectangles]
        all_diameters.extend(diameters)

        total_images += 1  # 统计总图片数量

# 创建一个图形和一个坐标轴
fig, ax = plt.subplots(figsize=(8, 6))

# 计算统计区域的体积（立方毫米）
volume = image_width * pixel_size * image_height * pixel_size * depth * 1e-9

# 计算时间（秒）
time = total_images / Fps

# 绘制频率直方图
hist, bins = np.histogram(all_diameters, bins=100, range=(1, 250), density=False)
# 计算液滴数密度
droplet_density_hist = hist / (time * volume)

# 计算核密度估计
kde = gaussian_kde(all_diameters)

# 生成用于绘制核密度图的数据点
x_eval = np.linspace(min(all_diameters), max(all_diameters), 200)
kde_vals = kde(x_eval)

# 计算液滴数密度
droplet_density_kde = kde_vals / (time * volume)

# 计算核密度图的最大值
max_kde_density = np.max(droplet_density_kde)

# 计算直方图的最大值
max_hist_density = np.max(droplet_density_hist)

# 计算缩放系数
scale_factor = max_hist_density / max_kde_density

# 创建一个图形和一个坐标轴，并设置图片比例为 1:1
plt.rcParams.update({'font.size': 18})  # 调整字体大小

# 创建一个图形和一个坐标轴，并设置图片比例为 1:1，同时调大图形大小
fig, ax = plt.subplots(figsize=(8, 8))  # 调大图形大小

# 绘制液滴数密度直方图
ax.hist(bins[:-1], bins, weights=droplet_density_hist, alpha=0.85, color='blue', label='')

# 绘制核密度图
ax.fill_between(x_eval, droplet_density_kde * scale_factor, alpha=0.25, color='blue', label='')

# 设置图形标题和标签
ax.set_title('', fontsize=18)  # 调整标题字体大小
ax.set_xlabel('Droplet Diameter (μm)', fontsize=18)  # 调整 x 轴标签字体大小
ax.set_ylabel('Droplet Number Density, n(d,t) (1/mm³·s)', fontsize=18)  # 调整 y 轴标签字体大小

# 设置坐标轴的范围，保持 x 轴和 y 轴的比例相同
x_min, x_max = 0, 220
y_min, y_max = 0, 4500
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# 调整坐标轴的宽高比，使其与图片比例匹配
ax.set_aspect((x_max - x_min) / (y_max - y_min))

# 设置图例，并调整图例字体大小
ax.legend(fontsize=14)

# 调整坐标轴刻度标签的字体大小
ax.tick_params(axis='both', labelsize=16)

# 显示图形
plt.show()