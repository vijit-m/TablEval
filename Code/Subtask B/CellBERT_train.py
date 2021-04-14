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

drive.mount("/content/Drive")




###############################################################
# load the preprocessed jsons of train and test files. Use the train to split for validation as well
###############################################################

path_to_train_file = "/content/Drive/MyDrive/home/subtaskBdata_json/subtaskb_training_all_2.json"
path_to_test_file = "/content/Drive/MyDrive/test_subtaskB.json"
path_to_save_model = "/content/Drive/MyDrive/home/subtaskBdata_json/BERT_small_subtaskb/"

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

data_json_train, data_json_test = load_jsons(path_to_test_file, path_to_train_file)




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

train_set, val_set = prepare_train_and_val_data(data_json_train, 500000, 50000)
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




validation_set = val_set
train_input_ids, train_lengths = input_id_maker(train_set, tokenizer)
validation_input_ids, validation_lengths = input_id_maker(validation_set, tokenizer)
test_input_ids, test_lengths = input_id_maker(test_set, tokenizer)

train_labels = label_modifier(train_set)
validation_labels = label_modifier(validation_set)
test_labels = label_modifier(test_set, "test")

train_attention_masks = att_masking(train_input_ids)
validation_attention_masks = att_masking(validation_input_ids)
test_attention_masks = att_masking(test_input_ids)

train_inputs = train_input_ids
validation_inputs = validation_input_ids
test_inputs = test_input_ids
train_masks = train_attention_masks
validation_masks = validation_attention_masks
test_masks = test_attention_masks

train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)
validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(validation_labels)
validation_masks = torch.tensor(validation_masks)
test_inputs = torch.tensor(test_inputs)
test_labels = torch.tensor(test_labels)
test_masks = torch.tensor(test_masks)

batch_size = 32
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size = batch_size)
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = RandomSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size = batch_size)
test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size = batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.to(device)




lr = 2e-5
max_grad_norm = 1.0
epochs = 5
num_total_steps = len(train_dataloader)*epochs
num_warmup_steps = 1000
warmup_proportion = float(num_warmup_steps) / float(num_total_steps)  # 0.1
optimizer = AdamW(model.parameters(), lr=lr, correct_bias=True)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = num_warmup_steps, num_training_steps = num_total_steps)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

seed_val = 2212


np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


loss_values = []





# For each epoch...
for epoch_i in range(0, epochs):
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()
    total_loss = 0

    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}. '.format(step, len(train_dataloader)))

        
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()        

        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)            
    loss_values.append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
        
    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():        
          outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    
        logits = outputs[0]

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        eval_accuracy += tmp_eval_accuracy

        nb_eval_steps += 1

    # Report the final accuracy for this validation run.
    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))

print("")
print("Training complete!")




output_dir = './cellbert/'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)


model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)



!cp -r './cellbert' path_to_save_model
