# 加载 loss_record 文件
# PyTorch
import json

# For plotting
import matplotlib.pyplot as plt
# For data preprocess
import numpy as np
import torch


def load_loss_record(file_path):
    with open(file_path, 'r') as f:
        loss_record = json.load(f)
    return loss_record


# 绘制学习曲线
def plot_all_curves(loss_records, titles, total_steps=25000):
    ''' Plot learning curves of multiple DNNs (train & dev loss) '''
    plt.figure(figsize=(12, 8))

    for i, (loss_record, title) in enumerate(zip(loss_records, titles)):
        # 生成开发损失数据的横坐标
        dev_steps = len(loss_record['dev'])
        x_2 = np.linspace(0, min(total_steps - 1, len(loss_record['dev']) - 1), dev_steps, dtype=int)

        # 绘制开发损失曲线
        plt.plot(x_2, loss_record['dev'][:total_steps], label=f'{title} dev', linestyle='-', linewidth=1.5)

    # 设置图形的y轴范围
    plt.ylim(0.0, 2.0)
    # 设置图形的x轴标签
    plt.xlabel('Training steps')
    # 设置图形的y轴标签
    plt.ylabel('Loss')
    # 设置图形的标题
    plt.title('Learning Curves of Different Models')
    # 添加图例以区分训练和开发损失曲线
    plt.legend()
    # 显示图形
    plt.grid(True)
    plt.show()

def plot_curve():
    file_path = 'loss_record.json'
    title = 'loss_record'
    plt.figure(figsize=(12, 8))
    total_steps = 25000
    loss_record = load_loss_record(file_path)

    x_1 = range(min(total_steps, len(loss_record['train'])))
    # 生成开发损失数据的横坐标
    dev_steps = len(loss_record['dev'])
    x_2 = np.linspace(0, min(total_steps - 1, len(loss_record['train']) - 1), dev_steps, dtype=int)

    # 绘制训练损失曲线
    if title == 'loss_record':
        plt.plot(x_1, loss_record['train'][:total_steps], label=f'{title} train', linestyle='--', linewidth=1.5)
    # 绘制开发损失曲线
    plt.plot(x_2, loss_record['dev'][:total_steps], label=f'{title} dev', linestyle='-', linewidth=1.5)

    # 设置图形的y轴范围
    plt.ylim(0.3, 1.0)
    # 设置图形的x轴标签
    plt.xlabel('Training steps')
    # 设置图形的y轴标签
    plt.ylabel('Loss')
    # 设置图形的标题
    plt.title('Learning Curves of Different Models')
    # 添加图例以区分训练和开发损失曲线
    plt.legend()
    # 显示图形
    plt.grid(True)
    plt.show()
# 文件路径
file_paths = [
    'loss_record_batch300.json',
    'loss_record_L2_1e-3_2layers64-32.json',
    'loss_record_layer48-16.json',
    'loss_record_layer64-16.json',
    'loss_record_layer96-16.json',
    'loss_record_lr0001.json',
    'loss_record_lr00015.json',
    'loss_record_ls00015.json',
    'loss_record073.json',
    'loss_record.json',
    'best_loss_record.json'
]

# 标题
titles = [
    'batch300',
    'L2_1e-3_2layers64-32',
    'layer48-16',
    'layer64-16',
    'layer96-16',
    'lr0001',
    'lr00015',
    'ls00015',
    'loss_record073',
    'loss_record',
    'best_loss_record'
]



# 加载 loss_record
loss_records = [load_loss_record(file_path) for file_path in file_paths]

# 绘制学习曲线
#plot_curve()
plot_all_curves(loss_records, titles)