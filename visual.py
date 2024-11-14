import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def visualize_attention(img, pos, attn, object_boxes=None):
    # img: 输入图像
    # pos: 采样点 (B*G, Hk, Wk, 2)
    # attn: 注意力权重 (heads, HW, Ns)
    # object_boxes: 目标边界框 (x_min, y_min, x_max, y_max)

    H, W, _ = img.shape
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    # 从 pos 获取 (x, y) 采样点
    pos_pixel = (pos + 1) * 0.5 * torch.tensor([H, W], device=pos.device).view(1, 1, 1, 2)

    pos_pixel = pos_pixel.cpu().detach().numpy()

    # 获取第一个 head 的注意力权重 (展示第一个 head 的注意力图)
    # attn_scores = attn[1].cpu().detach().numpy()
    attn_scores = attn.sum(dim=0).cpu().detach().numpy()
    # attn_scores = attn.mean(dim=0).cpu().detach().numpy()
    # 归一化注意力得分，用于调整圆圈大小
    attn_scores = (attn_scores - attn_scores.min()) / (attn_scores.max() - attn_scores.min())
    k=0
    h=attn.shape[1]** 0.5
    # r=387
    r=520
    # r=451
    # 遍历每个采样点并绘制圆圈
    for i in range(pos_pixel.shape[1]):  # 对应 height 维度
        for j in range(pos_pixel.shape[2]):  # 对应 width 维度
            x, y = pos_pixel[0, i, j, 1], pos_pixel[0, i, j, 0]

            # 使用 attn_scores[i, j] 提取标量值
            radius = float(attn_scores[500, k] * 10)  # 确保是标量
            if radius > 4.2:
             circle = plt.Circle((x, y), radius, color='blue', alpha=0.6)
             ax.add_patch(circle)
            k=k+1
    row = r // h  # 计算行索引
    col = r % h   # 计算列索引
    reference = plt.Circle((col* (512 / h), row* (512 / h)), 10, color='red', alpha=0.6)
    # ax.plot(col* (512 / H), row* (512 / W), marker='*', color='blue', markersize=10, alpha=0.8)
    ax.add_patch(reference)
    # 绘制目标框
    if object_boxes is not None:
        for box in object_boxes:
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    plt.show()

# 示例数据
# img = plt.imread('T5_0106_false_color.tif')  # 使用实际输入图像
# pos = torch.rand(16, 32, 32, 2)  # 使用真实的 pos 替换
# attn = torch.rand(64, 1024, 1024)  # 使用真实的 attn 替换
# object_boxes = None  # 示例目标框 (xmin, ymin, xmax, ymax)
#
# visualize_attention(img, pos, attn, object_boxes)
