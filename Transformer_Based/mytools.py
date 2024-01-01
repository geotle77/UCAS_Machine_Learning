import csv
from sklearn.metrics import f1_score

def get_test_accuracy(predicted_values, file_path):
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        standard_answers = [row[0] for row in reader]
        standard_answers.pop(0)
        standard_answers = [int(i) for i in standard_answers]

        total_samples = len(predicted_values)
        correct_predictions = sum(1 for pred, ans in zip(predicted_values, standard_answers) if pred == ans)
        accuracy = round((correct_predictions / total_samples)*100, 2)

        f1_s = round(f1_score(standard_answers, predicted_values, average='weighted')*100, 2)

        return accuracy, f1_s

def record_acc(train_acc_lst, test_acc_lst, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        # 创建一个CSV写入器
        csvwriter = csv.writer(csvfile)

        # 写入表头，这里包括两列：索引列和元素列
        csvwriter.writerow(['Epoch', 'train_acc', 'test_acc'])

        # 通过enumerate遍历列表，将索引和元素写入CSV文件
        for index, (elem1, elem2) in enumerate(zip(train_acc_lst, test_acc_lst)):
            csvwriter.writerow([index, elem1, elem2])

def record_f1_scores(f1_s_lst, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        # 创建一个CSV写入器
        csvwriter = csv.writer(csvfile)

        # 写入表头，这里包括两列：索引列和元素列
        csvwriter.writerow(['Epoch', 'F1_Scores'])

        # 通过enumerate遍历列表，将索引和元素写入CSV文件
        for index, element in enumerate(f1_s_lst):
            csvwriter.writerow([index, element])


def record_loss(f1_s_lst, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        # 创建一个CSV写入器
        csvwriter = csv.writer(csvfile)

        # 写入表头，这里包括两列：索引列和元素列
        csvwriter.writerow(['Epoch', 'Loss'])

        # 通过enumerate遍历列表，将索引和元素写入CSV文件
        for index, element in enumerate(f1_s_lst):
            csvwriter.writerow([index, element])
