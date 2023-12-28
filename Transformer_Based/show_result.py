import csv
import matplotlib.pyplot as plt

def show_acc(file_path):
    x, y1, y2 = [], [], []
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # 跳过表头
        for row in csvreader:
            x.append(float(row[0]))  # 第一列作为x轴
            y1.append(float(row[1]))  # 第二列作为y轴的第一个变量
            y2.append(float(row[2]))  # 第三列作为y轴的第二个变量

    # 绘制坐标图
    plt.plot(x, y1, label='train_acc')
    plt.plot(x, y2, label='test_acc')

    # 设置图形标题和坐标轴标签
    plt.title("ACCs changes with Epoch")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    # 添加图例
    plt.legend()

    # 显示图形
    plt.show()


def show_f1_scores(file_path):
    x, y = [], []
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # 跳过表头
        for row in csvreader:
            x.append(float(row[0]))  # 第一列作为x轴
            y.append(float(row[1]))  # 第二列作为y轴的变量

    # 绘制坐标图
    plt.plot(x, y, label='f1_scores')

    # 设置图形标题和坐标轴标签
    plt.title("F1_scores changes with Epoch")
    plt.xlabel('Epoch')
    plt.ylabel('F1_scores')

    # 添加图例
    plt.legend()

    # 显示图形
    plt.show()

if __name__ == '__main__':
    show_acc('Acc.csv')
    show_f1_scores('F1_Scores.csv')