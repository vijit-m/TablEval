####!pip install transformers
####!pip install sentencepiece
import os
import random
import pandas as pd
import numpy as np
import csv
import tensorflow as tf
import torch
from sklearn.model_selection import train_test_split
from google.colab import drive
import textwrap
import progressbar
import keras
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
import time
import datetime
import json
from google.colab import drive
import pandas as pd
import xml.etree.ElementTree as ET
import os

drive.mount("/content/Drive")




###############################################################
# load the preprocessed jsons of train and test files. Use the train to split for validation as well
###############################################################

path_to_test_file = "/content/Drive/MyDrive/test_subtaskB.json"
path_to_saved_model = "/content/Drive/MyDrive/home/subtaskBdata_json/BERT_small_subtaskb/"

def load_jsons(path_to_test_file, path_to_train_file=""):
  if path_to_train_file is not "":
    f1 = open(path_to_train_file, "r")
    data_train = json.load(f1)
    f1.close()
    f2 = open(path_to_test_file, "r")
    data_test = json.load(f2)
    f2.close()
    return data_train, data_test
  else:
    f2 = open(path_to_test_file, "r")
    data_test = json.load(f2)
    f2.close()
    return "", data_test

data_json_train, data_json_test = load_jsons(path_to_test_file, "")




def equalize_unbalanced_sets(df_train, df_val):
  df_train_pos = df_train[df_train['label']==2] 
  df_train_neg = df_train[df_train['label']==0]
  df_val_pos = df_val[df_val['label']==2]
  df_val_neg = df_val[df_val['label']==0]

  sample_size_val_neg = len(df_val_pos["label"].to_list())
  sample_size_train_neg = len(df_train_pos["label"].to_list())
  df_train_neg_new = df_train_neg.sample(sample_size_train_neg)
  df_val_neg_new = df_val_neg.sample(sample_size_val_neg)

  train_set = pd.concat([df_train_pos, df_train_neg_new])
  val_set = pd.concat([df_val_pos, df_val_neg_new])

  train_set = train_set.sample(len(train_set))
  val_set = val_set.sample(len(val_set))
  return train_set, val_set

def prepare_train_and_val_data(data_json_train, beam_size=1000000, size_validation=10000):
  d_cleaned = []
  data = data_json_train
  for d in data:
    if d[4] == " #  # ":
      continue
    d_cleaned.append(d)
  
  train_d = d_cleaned[:beam_size]
  val_d = d_cleaned[beam_size:beam_size + size_validation]
  train_premise = [d[4] for d in train_d]
  train_hypo = [d[6] for d in train_d]
  train_label = [int(d[5]) for d in train_d]
  validation_premise = [d[4] for d in val_d]
  validation_hypo = [d[6] for d in val_d]
  validation_label = [int(d[5]) for d in val_d]
  df_train = pd.DataFrame(list(zip(train_premise, train_hypo, train_label)), columns =['Premise', 'Hypothesis', 'label']) 
  df_val = pd.DataFrame(list(zip(validation_premise, validation_hypo, validation_label)), columns =['Premise', 'Hypothesis', 'label'])
  equal_train, equal_val = equalize_unbalanced_sets(df_train, df_val)
  return equal_train, equal_val
  

def prepare_test_data(data_json_test, percentage=1):
  data = data_json_test
  d_cleaned = []
  for d in data:
    d_cleaned.append(d)
  test_d = d_cleaned[:int(len(d_cleaned))]
  for d in test_d:
    print(d)
    break
  test_tab_name = [d[0] for d in test_d]
  test_premise = [d[2] for d in test_d]
  test_hypo = [d[4] for d in test_d]
  test_label = [int(d[3]) for d in test_d]
  df_test = pd.DataFrame(list(zip(test_tab_name, test_premise, test_hypo, test_label)), columns =['Table', 'Premise', 'Hypothesis', 'label'])
  test_set = df_test[:percentage*len(df_test)]
  return test_set

#train_set, val_set = prepare_train_and_val_data(data_json_train, 500000, 50000)
test_set = prepare_test_data(data_json_test, 1)







MODEL_CLASSES = {
    'bert': (BertForSequenceClassification, BertTokenizer, BertConfig),
    'xlnet': (XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig),
    'xlm': (XLMForSequenceClassification, XLMTokenizer, XLMConfig),
    'roberta': (RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig),
    'distilbert': (DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig)}

model_type = 'bert' ###--> CHANGE WHAT MODEL YOU WANT HERE!!! <--###
model_class, tokenizer_class, config_class = MODEL_CLASSES[model_type]
model_name = 'bert-base-uncased'
# tokenizer = tokenizer_class.from_pretrained("allenai/scibert_scivocab_uncased")
tokenizer = BertTokenizer.from_pretrained(path_to_saved_model)

def input_id_maker(dataf, tokenizer):
  input_ids = []
  lengths = []

  for i in progressbar.progressbar(range(len(dataf['Premise']))):
    sen_a = dataf['Premise'].iloc[i]
    sen_b = dataf['Hypothesis'].iloc[i]
    sen_a_toks = tokenizer.tokenize(sen_a)
    sen_b_toks = tokenizer.tokenize(sen_b)
    CLS = tokenizer.cls_token
    SEP = tokenizer.sep_token
    sen = [CLS] + sen_a_toks + [SEP] + sen_b_toks
    if(len(sen) > 127):
      sen = sen[:127]
    sen = sen + [SEP]
    encoded_sent = tokenizer.convert_tokens_to_ids(sen)
    input_ids.append(encoded_sent)
    lengths.append(len(encoded_sent))

  input_ids = pad_sequences(input_ids, maxlen=128, value=0, dtype="long", truncating="pre", padding="post")
  return input_ids, lengths

def att_masking(input_ids):
  attention_masks = []
  for sent in input_ids:
    att_mask = [int(token_id > 0) for token_id in sent]
    attention_masks.append(att_mask)
  return attention_masks

def label_modifier(df, data_type="train"):
  if(data_type == "test"):
    return np.array(df['label'].to_list())
  labels = df['label'].to_list()
  new_labels = []
  for l in labels:
    if(l==2):
      new_labels.append(int(1))
    else:
      new_labels.append(int(0))
  return np.array(new_labels)




model = BertForSequenceClassification.from_pretrained(path_to_saved_model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

test_input_ids, test_lengths = input_id_maker(test_set, tokenizer)
test_attention_masks = att_masking(test_input_ids)

batch_size = 32
test_labels = label_modifier(test_set, "test")

test_inputs = test_input_ids

test_masks = test_attention_masks

test_inputs = torch.tensor(test_inputs)
test_labels = torch.tensor(test_labels)
test_masks = torch.tensor(test_masks)


test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_dataloader = DataLoader(test_data, batch_size = batch_size)





print('Predicting labels for {:,} test sentences...'.format(len(test_inputs)))
model.eval()
predictions , true_labels = [], []

for (step, batch) in enumerate(test_dataloader):
  if(step%100==0):
    print(100*step/len(test_dataloader))
  batch = tuple(t.to(device) for t in batch)
  b_input_ids, b_input_mask, b_labels = batch
  
  with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs = model(b_input_ids, token_type_ids=None, 
                      attention_mask=b_input_mask)

  logits = outputs[0]
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()
  
  predictions.append(logits)
  true_labels.append(label_ids)





predictions_concat = np.concatenate(predictions, axis=0)
true_labels_concat = np.concatenate(true_labels, axis=0)
pred_flat = np.argmax(predictions_concat, axis=1).flatten()
labels_flat = true_labels_concat.flatten()
converted_label_predictions = []
for p in pred_flat:
  if(p==1):
    converted_label_predictions.append(2)
  else:
    converted_label_predictions.append(0)

test_set["pred_label"] = converted_label_predictions





df = test_set
preds_list = df["pred_label"].to_list()

for fn in os.listdir("dev_v1.1/v1.1/input/"):
    if(fn[0]!="1"):
        continue
    path = "dev_v1.1/v1.1/input/" + fn
    f = open(path, "r", encoding="utf8")
    tree = ET.parse(f)
    root = tree.getroot()
    table_list=[]
    for child in root:
        table_list.append(child.attrib['id'][6:])
    
    for i in range(len(table_list)):
          for cell in root[i].iter('cell'):
            for line in cell.iter('evidence'):
                line.attrib['version'] = '0'
                if(preds_list[count]==2):
                    line.attrib['type'] = "relevant"
                else:
                    line.attrib['type'] = "irrelevant"
                
    f.close()
    tree.write(fn)