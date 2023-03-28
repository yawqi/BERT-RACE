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
import torch
from CustomLabelAccuracyEvaluator import CustomLabelAccuracyEvaluator

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

#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-uncased'
# Read the dataset
train_batch_size = 32
num_epochs = 3

task_1_data_dir = './RACE-SR-NEW'
curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
task_1_model_save_path = 'output-1-(3->1)/task-1-'+model_name.replace("/", "-")+'-'+ curr_time
# task_1_model_save_path = 'output-1/task-1-'+model_name.replace("/", "-")+'-'+ curr_time
# task_2_model_save_path = 'output-2/task-2-'+model_name.replace("/", "-")+'-'+ curr_time
# task_3_model_save_path = 'output-3/task-3-'+model_name.replace("/", "-")+'-'+ curr_time

num_labels = 5
device_name = "cuda:0"
froze_params = False
train_samples = read_data_from_path(os.path.join(task_1_data_dir, 'train'), max_label=num_labels)
dev_samples = read_data_from_path(os.path.join(task_1_data_dir, 'dev'), max_label=num_labels)

# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name)
for param in word_embedding_model.parameters():
    param.requires_grad = not froze_params
# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device_name)

# Convert the dataset to a DataLoader ready for training
logging.info("Read Task1 train dataset")

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
dev_dataloader = DataLoader(dev_samples, shuffle=True, batch_size=train_batch_size)

train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=num_labels)
train_loss.to(device_name)
logging.info("Read Task-1 dev dataset")
evaluator = CustomLabelAccuracyEvaluator(dev_dataloader, softmax_model=train_loss, name='task1-dev', device=device_name)

# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_dataloader) * num_epochs  * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=2500,
          warmup_steps=warmup_steps,
          output_path=task_1_model_save_path)

softmax_loss_path = os.path.join(task_1_model_save_path, "softmax_loss.pth")
torch.save(train_loss.classifier.state_dict(), softmax_loss_path)

##############################################################################
#
# Load the stored model and evaluate its performance on task1 benchmark dataset
#
##############################################################################

test_samples = read_data_from_path(os.path.join(task_1_data_dir, 'test'), max_label=num_labels)
model = SentenceTransformer(task_1_model_save_path, device=device_name)
test_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=num_labels)
test_loss.classifier.load_state_dict(torch.load(softmax_loss_path))
test_loss.to(device_name)
test_dataloader = DataLoader(test_samples, shuffle=True, batch_size=train_batch_size)
test_evaluator = CustomLabelAccuracyEvaluator(test_dataloader, name='task1-test', softmax_model=train_loss, device=device_name)
test_evaluator(model, output_path=task_1_model_save_path)
