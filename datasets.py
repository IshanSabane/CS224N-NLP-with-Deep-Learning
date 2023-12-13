#!/usr/bin/env python3

'''
This module contains our Dataset classes and functions to load the 3 datasets we're using.

You should only need to call load_multitask_data to get the training and dev examples
to train your model.
'''


import csv
import pandas as pd
import torch
from torch.utils.data import Dataset
from tokenizer import BertTokenizer
import json
from collections import defaultdict
import random
# import torchtext

def preprocess_string(s):
    return ' '.join(s.lower()
                    .replace('.', ' .')
                    .replace('?', ' ?')
                    .replace(',', ' ,')
                    .replace('\'', ' \'')
                    .split())


class SentenceClassificationDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):

        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        labels = torch.LongTensor(labels)

        return token_ids, attention_mask, labels, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, labels, sents, sent_ids= self.pad_data(all_data)

        batched_data = {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'sents': sents,
                'sent_ids': sent_ids
            }

        return batched_data


class SentenceClassificationTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        sent_ids = [x[1] for x in data]

        encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        return token_ids, attention_mask, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, sents, sent_ids= self.pad_data(all_data)

        batched_data = {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'sents': sents,
                'sent_ids': sent_ids
            }

        return batched_data


class SentencePairDataset(Dataset):
    def __init__(self, dataset, args, isRegression =False, join_sentences = False):
        self.dataset = dataset
        self.p = args
        self.isRegression = isRegression
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.join_sentences = join_sentences
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):

        # if self.join_sentences == True: 
        #     sent1 = [ x[0]   + " [SEP] " + x[1] for x in data]
        #     sent2 = [ x[1]   + " [SEP] " + x[0] for x in data]
            
        #     # print(sent1)
            
        #     # encoding1 = self.tokenizer( text_pair=[[x[0], x[1]] for x in data], return_tensors='pt', padding = True, truncation = True)
        #     # encoding2 = self.tokenizer(text_pair=[[x[1], x[0]] for x in data], return_tensors='pt', padding = True, truncation = True)
            
        #     encoding1 = self.tokenizer(sent1 , return_tensors='pt', padding = True, truncation = True)
        #     encoding2 = self.tokenizer(sent2 , return_tensors='pt', padding = True, truncation = True)
            
        # else:

        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]

        labels = [x[2] for x in data]
        sent_ids = [x[3] for x in data]

        encoding1 = self.tokenizer(sent1, return_tensors='pt', padding=True, truncation=True)
        encoding2 = self.tokenizer(sent2, return_tensors='pt', padding=True, truncation=True)



        token_ids = torch.LongTensor(encoding1['input_ids'])
        attention_mask = torch.LongTensor(encoding1['attention_mask'])
        token_type_ids = torch.LongTensor(encoding1['token_type_ids'])


        token_ids2 = torch.LongTensor(encoding2['input_ids'])
        attention_mask2 = torch.LongTensor(encoding2['attention_mask'])
        token_type_ids2 = torch.LongTensor(encoding2['token_type_ids'])

        if self.isRegression:
            labels = torch.DoubleTensor(labels)
        else:
            labels = torch.LongTensor(labels)
            

        return (token_ids, token_type_ids, attention_mask,
                token_ids2, token_type_ids2, attention_mask2,
                labels,sent_ids)

    def collate_fn(self, all_data):
        (token_ids, token_type_ids, attention_mask,
         token_ids2, token_type_ids2, attention_mask2,
         labels, sent_ids) = self.pad_data(all_data)

        batched_data = {
                'token_ids_1': token_ids,
                'token_type_ids_1': token_type_ids,
                'attention_mask_1': attention_mask,
                'token_ids_2': token_ids2,
                'token_type_ids_2': token_type_ids2,
                'attention_mask_2': attention_mask2,
                'labels': labels,
                'sent_ids': sent_ids
            }

        return batched_data


class SentencePairTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.join_sentences = False
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        
        
        # if self.join_sentences == True: 
        #     sent1 = [ x[0]   + " [SEP] " + x[1] for x in data]
        #     sent2 = [ x[1]   + " [SEP] " + x[0] for x in data]
            
        #     # encoding1 = self.tokenizer([[x[0], x[1]] for x in data], return_tensors='pt', padding = True, truncation = True)
        #     # encoding2 = self.tokenizer([[x[1], x[0]] for x in data], return_tensors='pt', padding = True, truncation = True)
        

        #     encoding1 = self.tokenizer(sent1, return_tensors='pt', padding = True, truncation = True)
        #     encoding2 = self.tokenizer(sent2, return_tensors='pt', padding = True, truncation = True)
        

        # else:
        

        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding1 = self.tokenizer(sent1, return_tensors='pt', padding=True, truncation=True)
        encoding2 = self.tokenizer(sent2, return_tensors='pt', padding=True, truncation=True)

       
        token_ids = torch.LongTensor(encoding1['input_ids'])
        attention_mask = torch.LongTensor(encoding1['attention_mask'])
        token_type_ids = torch.LongTensor(encoding1['token_type_ids'])

        token_ids2 = torch.LongTensor(encoding2['input_ids'])
        attention_mask2 = torch.LongTensor(encoding2['attention_mask'])
        token_type_ids2 = torch.LongTensor(encoding2['token_type_ids'])


        return (token_ids, token_type_ids, attention_mask,
                token_ids2, token_type_ids2, attention_mask2,
               sent_ids)

    def collate_fn(self, all_data):
        (token_ids, token_type_ids, attention_mask,
         token_ids2, token_type_ids2, attention_mask2,
         sent_ids) = self.pad_data(all_data)

        batched_data = {
                'token_ids_1': token_ids,
                'token_type_ids_1': token_type_ids,
                'attention_mask_1': attention_mask,
                'token_ids_2': token_ids2,
                'token_type_ids_2': token_type_ids2,
                'attention_mask_2': attention_mask2,
                'sent_ids': sent_ids
            }

        return batched_data


def load_multitask_test_data():
    paraphrase_filename = f'data/quora-test.csv'
    sentiment_filename = f'data/ids-sst-test.txt'
    similarity_filename = f'data/sts-test.csv'

    sentiment_data = []

    with open(sentiment_filename, 'r') as fp:
        for record in csv.DictReader(fp,delimiter = '\t'):
            sent = record['sentence'].lower().strip()
            sentiment_data.append(sent)

    print(f"Loaded {len(sentiment_data)} test examples from {sentiment_filename}")

    paraphrase_data = []
    with open(paraphrase_filename, 'r') as fp:
        for record in csv.DictReader(fp,delimiter = '\t'):
            #if record['split'] != split:
            #    continue
            paraphrase_data.append((preprocess_string(record['sentence1']),
                                    preprocess_string(record['sentence2']),
                                    ))

    print(f"Loaded {len(paraphrase_data)} test examples from {paraphrase_filename}")

    similarity_data = []
    with open(similarity_filename, 'r') as fp:
        for record in csv.DictReader(fp,delimiter = '\t'):
            similarity_data.append((preprocess_string(record['sentence1']),
                                    preprocess_string(record['sentence2']),
                                    ))

    print(f"Loaded {len(similarity_data)} test examples from {similarity_filename}")

    return sentiment_data, paraphrase_data, similarity_data



def load_multitask_data(sentiment_filename,paraphrase_filename,similarity_filename,split='train'):
    sentiment_data = []
    num_labels = {}
    if split == 'test':
        with open(sentiment_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                sentiment_data.append((sent,sent_id))
    else:
        with open(sentiment_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                label = int(record['sentiment'].strip())
                if label not in num_labels:
                    num_labels[label] = len(num_labels)
                sentiment_data.append((sent, label,sent_id))

    print(f"Loaded {len(sentiment_data)} {split} examples from {sentiment_filename}")

    paraphrase_data = []
    if split == 'test':
        with open(paraphrase_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent_id = record['id'].lower().strip()
                paraphrase_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2']),
                                        sent_id))

    else:
        with open(paraphrase_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                try:
                    sent_id = record['id'].lower().strip()
                    paraphrase_data.append((preprocess_string(record['sentence1']),
                                            preprocess_string(record['sentence2']),
                                            int(float(record['is_duplicate'])),sent_id))
                except:
                    pass

    print(f"Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")

    similarity_data = []
    if split == 'test':
        with open(similarity_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent_id = record['id'].lower().strip()
                similarity_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2'])
                                        ,sent_id))
    else:
        with open(similarity_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent_id = record['id'].lower().strip()
                similarity_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2']),
                                        float(record['similarity']),sent_id))

    print(f"Loaded {len(similarity_data)} {split} examples from {similarity_filename}")

    return sentiment_data, num_labels, paraphrase_data, similarity_data



def load_snli(split = 'train'):

    # df_train = pd.read_csv('./data/snli_1.0_train.txt', sep='\t')[[ 'pairID','gold_label','sentence1','sentence2']]
    # df_dev = pd.read_csv('./data/snli_1.0_dev.txt', sep='\t')[['pairID','gold_label','sentence1','sentence2']]
    # df_test = pd.read_csv('./data/snli_1.0_test.txt', sep='\t')[['pairID','gold_label','sentence1','sentence2']]


    df = pd.read_csv(f'./data/snli_1.0_{split}.txt', sep='\t')[[ 'pairID','gold_label','sentence1','sentence2']]
    # df = list(zip(df.sentence1, df.sentence2, df.gold_label, df.pairID))



    # df_train = list(zip(df_train.sentence1, df_train.sentence2, df_train.gold_label, df_train.pairID))
    # df_dev = list(zip(df_dev.sentence1, df_dev.sentence2, df_dev.gold_label, df_dev.pairID))
    # df_test = list(zip(df_test.sentence1, df_test.sentence2, df_test.gold_label, df_test.pairID))

   
    # Load SNLI dataset
    # text_field = torchtext.data.Field(tokenize='spacy')
    # label_field = torchtext.data.Field(sequential=False)
    # train_data, _, _ = torchtext.datasets.SNLI.splits(text_field, label_field)

    # # Filter dataset to entailment and contradiction labels
    train_data = df[ (df.gold_label == 'entailment') | (df.gold_label =='contradiction')]
    # train_data = df.filter(lambda example: example.gold_label in ['entailment', 'contradiction'])

    # # # Create sentence triplets
    triplets  = dict()
    for i in list(train_data.sentence1):
        triplets[i]={}
    
    for index, example in train_data.iterrows():
        
        triplets[example.sentence1][example.gold_label] = example.sentence2 


    data = [(key, value.get('entailment',''), value.get('contradiction','') ) for key,value in triplets.items() ]

    return data


class SNLI(Dataset):
    def __init__(self, dataset, args, isRegression =False, join_sentences = False):
        self.dataset = dataset
        self.p = args
        self.isRegression = isRegression
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.join_sentences = join_sentences
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):

        anchor = [x[0] for x in data]
        postive = [x[1] for x in data]
        negative = [x[2] for x in data]
        
        # labels = [x[3] for x in data]
        # sent_ids = [x[4] for x in data]


        anchor_encoding = self.tokenizer(anchor, return_tensors='pt', padding=True, truncation=True)
        positive_encoding = self.tokenizer(postive, return_tensors='pt', padding=True, truncation=True)
        negative_encoding = self.tokenizer(negative, return_tensors='pt', padding=True, truncation=True)

        anchor_token_ids = torch.LongTensor(anchor_encoding['input_ids'])
        anchor_attention_mask = torch.LongTensor(anchor_encoding['attention_mask'])
        anchor_token_type_ids = torch.LongTensor(anchor_encoding['token_type_ids'])


        positve_token_ids = torch.LongTensor(positive_encoding['input_ids'])
        positve_attention_mask = torch.LongTensor(positive_encoding['attention_mask'])
        positve_token_type_ids = torch.LongTensor(positive_encoding['token_type_ids'])

        negative_token_ids = torch.LongTensor(negative_encoding['input_ids'])
        negative_attention_mask = torch.LongTensor(negative_encoding['attention_mask'])
        negative_token_type_ids = torch.LongTensor(negative_encoding['token_type_ids'])




        # if self.isRegression:
        #     labels = torch.DoubleTensor(labels)
        # else:
        #     labels = torch.LongTensor(labels)
            

        return (anchor_token_ids, anchor_attention_mask , anchor_token_type_ids,
               positve_token_ids, positve_attention_mask , positve_token_type_ids,
                negative_token_ids, negative_attention_mask , negative_token_type_ids)
                # labels,sent_ids)

    def collate_fn(self, all_data):
        (anchor_token_ids, anchor_attention_mask , anchor_token_type_ids,
         positve_token_ids, positve_attention_mask , positve_token_type_ids,
         negative_token_ids, negative_attention_mask ,
         negative_token_type_ids) = self.pad_data(all_data)

        batched_data = {
                'anchor_token_ids': anchor_token_ids,
                'anchor_attention_mask': anchor_attention_mask,
                'anchor_token_type_ids': anchor_token_type_ids,
                'positve_token_ids': positve_token_ids,
                'positve_attention_mask': positve_attention_mask,
                'positve_token_type_ids': positve_token_type_ids,
                'negative_token_ids': negative_token_ids,
                'negative_attention_mask': negative_attention_mask,
                'negative_token_type_ids': negative_token_type_ids,
                # 'labels': labels,
                # 'sent_ids': sent_ids
            }

        return batched_data
