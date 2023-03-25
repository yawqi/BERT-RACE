from sentence_transformers.evaluation import SentenceEvaluator
import torch
from torch.utils.data import DataLoader
import logging
import os
import csv
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses
from sentence_transformers.readers import InputExample
import logging
import os
import glob
import re
import json
from sklearn.metrics import classification_report
import torch
import CustomLabelAccuracyEvaluator

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

def read_data_from_path(path, max_label):
    dirs = glob.glob(path+"/C*")
    examples = []
    for d in dirs:
        print("*** Scanning dir %s ***" % d)
        match = re.search(r'C(\d+)', d)
        label = int(match.group(1)) - 1
        if max_label and max_label - 1 < label:
            continue
        label = max_label - 1 - label
        filenames = glob.glob(d + '/*')
        for filename in filenames:
            print("Open file %s" % filename)
            with open(filename, 'r', encoding='utf-8') as fpr:
                # pid = int(match.group(1))
                data_raw = json.load(fpr)
                examples.append(InputExample(texts=[data_raw['s1'], data_raw['s2']], label=label))
                print('label:{}'.format(label))
    return examples

def batch_to_device(batch, target_device: torch.device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(target_device)
    return batch

class CustomLabelAccuracyEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on its accuracy on a labeled dataset

    This requires a model with LossFunction.SOFTMAX

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(self, dataloader: DataLoader, device, name: str = "", softmax_model = None, write_csv: bool = True):
        """
        Constructs an evaluator for the given dataset

        :param dataloader:
            the data for the evaluation
        """
        self.dataloader = dataloader
        self.device = device
        self.name = name
        self.softmax_model = softmax_model

        if name:
            name = "_"+name

        self.write_csv = write_csv
        self.csv_file = "accuracy_evaluation"+name+"_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy"]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = 0) -> float:
        model.eval()
        total = 0
        correct = 0
        true_labels = []
        predicted_labels = []

        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logging.info("Evaluation on the "+self.name+" dataset"+out_txt)
        self.dataloader.collate_fn = model.smart_batching_collate
        for step, batch in enumerate(self.dataloader):
            logging.info("step {}/{}".format(step, len(self.dataloader)))
            features, label_ids = batch
            label_ids = label_ids.to(self.device)
            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], self.device)
            with torch.no_grad():
                _, prediction = self.softmax_model(features, labels=None)

            total += prediction.size(0)
            correct += torch.argmax(prediction, dim=1).eq(label_ids).sum().item()
            true_labels.extend(label_ids.tolist())
            # predicted_labels.extend(prediction.tolist())
            predicted_labels.extend(torch.argmax(prediction, dim=1).tolist())
        logging.info("Evaluation over")
        accuracy = correct/total
        report = classification_report(true_labels, predicted_labels, digits=4, zero_division=0)
        report_dict = classification_report(true_labels, predicted_labels, digits=4, zero_division=0, output_dict=True)
        
        logging.info("Accuracy: {:.4f} ({}/{})\n".format(accuracy, correct, total))
        logging.info(f"{self.name} Classification Report:\n{report}")
        header = ["Class", "Precision", "Recall", "F1-Score", "Support"]
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline='', mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, accuracy])
            else:
                with open(csv_path, newline='', mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, accuracy])

            report_path = os.path.join(output_path, 'classification_report.csv')
            del report_dict["macro avg"]
            del report_dict["weighted avg"]
            if not os.path.isfile(report_path):
                with open(report_path, newline='', mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    # 写入标题行
                    header = ["Class", "Precision", "Recall", "F1-Score", "Support"]
                    writer.writerow(header)
                    # 写入各类别的指标
                    for class_label, metrics in report_dict.items():
                        if isinstance(metrics, float):
                            continue
                        row = [class_label, metrics["precision"], metrics["recall"], metrics["f1-score"], metrics["support"]]
                        writer.writerow(row)
            else:
                with open(report_path, newline='', mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    # 写入各类别的指标
                    for class_label, metrics in report_dict.items():
                        # if metrics not dict
                        if isinstance(metrics, float):
                            continue
                        row = [class_label, metrics["precision"], metrics["recall"], metrics["f1-score"], metrics["support"]]
                        writer.writerow(row)
        return accuracy

if __name__ == '__main__':
    task2 = False
    train_batch_size = 32

    task_data_dir = './RACE-SR-NEW'
    task_model_save_path = 'output-1/task-1-bert-base-uncased-2023-03-21_12-57-48'
    task_model_out_path = 'wq-test/'
    num_labels = 5
    device_name = "cuda:1"
    if task2:
        num_labels = 2
        task_data_dir = './TASK-2-DATA'
        task_model_save_path = './output-2/task-2-.-output-1-task-1-bert-base-uncased-2023-03-21_12-57-48-2023-03-22_11-19-35'

    model = SentenceTransformer(task_model_save_path, device=device_name)
    test_losses = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=num_labels)
    device = torch.device(device_name)
    test_losses.classifier.to(device)
    model.to(device)
    logging.info("Read Task Test dataset")
    test_samples = read_data_from_path(os.path.join(task_data_dir, 'test'), max_label=num_labels)[:1000]
    test_dataloader = DataLoader(test_samples, shuffle=True, batch_size=train_batch_size)
    test_evaluator = CustomLabelAccuracyEvaluator(test_dataloader, device, name='task-test-alone', softmax_model=test_losses)
    test_evaluator(model, output_path=task_model_save_path)
    # combined_state_dict = {
    #     'model_state_dict': model.state_dict(),
    #     'softmax_loss_state_dict': test_losses.state_dict()
    # }
    # torch.save(combined_state_dict, os.path.join(task_model_out_path, 'model_and_softmax_loss.pth'))
    # combined_state_dict = torch.load(os.path.join(task_model_out_path, 'model_and_softmax_loss.pth'))
    # model.load_state_dict(combined_state_dict['model_state_dict'])
    # test_losses.load_state_dict(combined_state_dict['softmax_loss_state_dict'])
