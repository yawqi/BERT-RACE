from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from sentence_transformers.evaluation import LabelAccuracyEvaluator
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import sys
import os
import glob
import re
import json
from sklearn.metrics import classification_report
import torch
import CustomLabelAccuracyEvaluator

#### Just some code to print debug information to stdout
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

model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-uncased'
train_batch_size = 32

task_1_data_dir = './RACE-SR-NEW'
curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
task_1_model_save_path = 'output-1/task-1-bert-base-uncased-2023-03-21_12-57-48'

num_labels = 5
device_name = "cuda:1"


# Convert the dataset to a DataLoader ready for training
logging.info("Read Task1 Test dataset")


##############################################################################
#
# Load the stored model and evaluate its performance on task1 benchmark dataset
#
##############################################################################

test_samples = read_data_from_path(os.path.join(task_1_data_dir, 'test'), max_label=num_labels)
model = SentenceTransformer(task_1_model_save_path, device=device_name)
test_losses = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=num_labels)
test_dataloader = DataLoader(test_samples, shuffle=True, batch_size=train_batch_size)
test_evaluator = CustomLabelAccuracyEvaluator(test_dataloader, name='task1-test', softmax_model=test_losses)
test_evaluator(model, output_path=task_1_model_save_path)