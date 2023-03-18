from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import sys
import os
import glob
import re
import json

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

def read_data_from_path(path):
    dirs = glob.glob(path+"/C*")
    examples = []
    for d in dirs:
        print("*** Scanning dir %s ***" % d)
        match = re.search(r'C(\d+)', d)
        label = int(match.group(1)) - 1
        filenames = glob.glob(d + '/*')
        for filename in filenames:
            print("Open file %s" % filename)
            with open(filename, 'r', encoding='utf-8') as fpr:
                # pid = int(match.group(1))
                data_raw = json.load(fpr)
                examples.append(InputExample(texts=[data_raw['s1'], data_raw['s2']], label=label))
    return examples

#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base
model_name = sys.argv[1] if len(sys.argv) > 1 else 'bert-base-uncased'
# Read the dataset
train_batch_size = 16
num_epochs = 4

curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
task_1_model_save_path = 'output-1/task-1-'+model_name.replace("/", "-")+'-'+ curr_time
task_2_model_save_path = 'output-2/task-2-'+model_name.replace("/", "-")+'-'+ curr_time
task_3_model_save_path = 'output-3/task-3-'+model_name.replace("/", "-")+'-'+ curr_time
task_1_data_dir = './RACE-SR'

num_labels = 6
train_samples = read_data_from_path(os.path.join(task_1_data_dir, 'train'))
dev_samples = read_data_from_path(os.path.join(task_1_data_dir, 'dev'))
test_samples = read_data_from_path(os.path.join(task_1_data_dir, 'test'))

# Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
word_embedding_model = models.Transformer(model_name)
# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device="cuda:1")

# Convert the dataset to a DataLoader ready for training
logging.info("Read Task1 train dataset")

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=num_labels)

logging.info("Read Task-1 dev dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='task1-dev')

# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_dataloader) * num_epochs  * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=task_1_model_save_path)

##############################################################################
#
# Load the stored model and evaluate its performance on task1 benchmark dataset
#
##############################################################################

model = SentenceTransformer(task_1_model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='task1-test')
test_evaluator(model, output_path=task_1_model_save_path)
