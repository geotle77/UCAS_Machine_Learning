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