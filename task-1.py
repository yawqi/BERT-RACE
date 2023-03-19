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

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

def read_data_from_path(path, max_label = None):
    dirs = glob.glob(path+"/C*")
    examples = []
    for d in dirs:
        print("*** Scanning dir %s ***" % d)
        match = re.search(r'C(\d+)', d)
        label = int(match.group(1)) - 1
        if max_label and max_label - 1 < label:
            continue
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
num_epochs = 2

task_1_data_dir = './RACE-SR'
curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
task_1_model_save_path = 'output-1/task-1-'+model_name.replace("/", "-")+'-'+ curr_time
task_2_model_save_path = 'output-2/task-2-'+model_name.replace("/", "-")+'-'+ curr_time
task_3_model_save_path = 'output-3/task-3-'+model_name.replace("/", "-")+'-'+ curr_time

num_labels = 6
train_samples = read_data_from_path(os.path.join(task_1_data_dir, 'train'))[::4]
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
dev_dataloader = DataLoader(dev_samples, shuffle=True, batch_size=train_batch_size)

train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=num_labels)

logging.info("Read Task-1 dev dataset")
evaluator = LabelAccuracyEvaluator(dev_dataloader, softmax_model=train_loss, name='task1-dev')

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
test_evaluator = LabelAccuracyEvaluator(test_samples, name='task1-test', softmax_model=train_loss)
test_evaluator(model, output_path=task_1_model_save_path)

##############################################################################
#
# Task 2
#
##############################################################################
# import torch
# from torch import nn, Tensor
# from typing import Iterable, Dict

# class CoSentLoss(nn.Module):
#     """
#     CosineSimilarityLoss expects, that the InputExamples consists of two texts and a float label.

#     It computes the vectors u = model(input_text[0]) and v = model(input_text[1]) and measures the cosine-similarity between the two.
#     By default, it minimizes the following loss: ||input_label - cos_score_transformation(cosine_sim(u,v))||_2.

#     :param model: SentenceTranformer model
#     :param loss_fct: Which pytorch loss function should be used to compare the cosine_similartiy(u,v) with the input_label? By default, MSE:  ||input_label - cosine_sim(u,v)||_2
#     :param cos_score_transformation: The cos_score_transformation function is applied on top of cosine_similarity. By default, the identify function is used (i.e. no change).

#     Example::

#             from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses

#             model = SentenceTransformer('distilbert-base-nli-mean-tokens')
#             train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
#                 InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.3)]
#             train_dataset = SentencesDataset(train_examples, model)
#             train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
#             train_loss = losses.CosineSimilarityLoss(model=model)


#     """
#     def __init__(self, model: SentenceTransformer):
#         super(CoSentLoss, self).__init__()
#         self.model = model
#         # self.cos_score_transformation = cos_score_transformation

#     def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
#         embeddings = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
#         output = self.cos_score_transformation(torch.cosine_similarity(embeddings[0], embeddings[1]))
#         return self.loss_fct(output, labels.view(-1))

# train_samples = read_data_from_path(os.path.join(task_1_data_dir, 'train'), max_label = 2)[::3]
# dev_samples = read_data_from_path(os.path.join(task_1_data_dir, 'dev'), max_label = 2)
# test_samples = read_data_from_path(os.path.join(task_1_data_dir, 'test'), max_label= 2)

# word_embedding_model_2 = models.Transformer(task_1_model_save_path)
# model = SentenceTransformer(modules=[word_embedding_model_2,])

# train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
# train_loss = losses.SoftmaxLoss(
#     model=model,
#     sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
#     num_labels=num_labels,
#     concatenation_sent_rep = True,
#     concatenation_sent_difference = True
# )

# logging.info("Read Task-2 dev dataset")
# # evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='task2-dev')

# # Configure the training. We skip evaluation in this example
# warmup_steps = math.ceil(len(train_dataloader) * num_epochs  * 0.1) #10% of train data for warm-up
# logging.info("Warmup-steps: {}".format(warmup_steps))

# # Train the model
# model.fit(train_objectives=[(train_dataloader, train_loss)],
#         #   evaluator=evaluator,
#           epochs=num_epochs,
#           evaluation_steps=1000,
#           warmup_steps=warmup_steps,
#           output_path=task_2_model_save_path)

# ##############################################################################
# #
# # Load the stored model and evaluate its performance on task1 benchmark dataset
# #
# ##############################################################################

# model = SentenceTransformer(task_2_model_save_path)
# test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='task2-test')
# test_evaluator(model, output_path=task_2_model_save_path)