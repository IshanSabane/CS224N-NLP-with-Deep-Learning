import time, random, numpy as np, argparse, sys, re, os
from types import SimpleNamespace
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm
from tokenizer import BertTokenizer

from datasets import SentenceClassificationDataset, SentencePairDataset, \
    load_multitask_data, load_multitask_test_data, load_snli,SNLI
from evaluation import model_eval_sst, test_model_multitask,model_eval_sts, model_eval_para
from pcgrad import PCGrad
best_dev = 0
TQDM_DISABLE=False




# fix the random seed
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5



# Implementation of Contrastive Loss. Snippets taken from the official SimCSE github repository
class TemperatureLoss(torch.nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, anchor_embedding, positive_embedding, negative_embedding):
        # Compute cosine similarity scores
        similarity_pos = F.cosine_similarity(anchor_embedding, positive_embedding, dim=-1) / self.temperature
        similarity_neg = F.cosine_similarity(anchor_embedding, negative_embedding, dim=-1) / self.temperature
        
        # Compute temperature-scaled contrastive loss with normalization
        logits = torch.cat([similarity_pos.unsqueeze(1), similarity_neg.unsqueeze(1)], dim=1)
        logits /= torch.std(logits)
        labels = torch.zeros(anchor_embedding.size(0), dtype=torch.long).to(device= torch.device('cuda'))
        loss = F.cross_entropy(logits, labels)
        
        return loss



class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp




class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:
    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        # You will want to add layers here to perform the downstream tasks.
        # Pretrain mode does not require updating bert paramters.
        
        
        
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        
        
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        

        # self.common = nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE, bias = True)
        self.common = nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE, bias = True)
        
        # the input sentence is of size BERT_HIDDEN_SIZE
        # the output is one of N_SENTIMENT_CLASSES classes
        self.sst_classifier = nn.Sequential(self.common, nn.Tanh(), 
            nn.Linear(BERT_HIDDEN_SIZE, 2 * BERT_HIDDEN_SIZE, bias = True),
            nn.Linear(2 * BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE, bias = True),
            nn.Linear(BERT_HIDDEN_SIZE, BERT_HIDDEN_SIZE // 2, bias = True),
            nn.Linear(BERT_HIDDEN_SIZE // 2, N_SENTIMENT_CLASSES, bias = True),
        )
        
        # we have two sentences which we check for paraphrasing
        # each sentence is of size BERT_HIDDEN_SIZE
        # the output is binary (yes/no) but the predict function outputs a single logit
        self.paraphrase_classifier1 = nn.Sequential( self.common, nn.Tanh(), 
            nn.Linear(BERT_HIDDEN_SIZE, 1, bias = True), nn.ReLU()
        )
        
        self.paraphrase_classifier2 = nn.Sequential( self.common,
            nn.Linear(BERT_HIDDEN_SIZE, 1, bias = True),nn.ReLU()
        )
        
        self.paraphrase_classifier3 = nn.Sequential( self.common,
            nn.Linear(BERT_HIDDEN_SIZE, 1, bias = True),nn.ReLU()
        )
        
        self.paraphrase_classifier_head = nn.Sequential(
            nn.Linear(3, 1, bias = True)
        ) 


        # we have two sentences which we check for similarity
        # each sentence is of size BERT_HIDDEN_SIZE
        # the output is one of 6 classes 
        # self.sts_classifier = nn.Sequential(
        #     nn.Linear(3 * BERT_HIDDEN_SIZE, 1)
        # )
        self.sts_classifier1 = nn.Sequential( self.common, nn.Tanh(), 
            nn.Linear(BERT_HIDDEN_SIZE, 1, bias = True), nn.ReLU()
        )
        self.sts_classifier2 = nn.Sequential( self.common,
            nn.Linear(BERT_HIDDEN_SIZE, 1, bias = True), nn.ReLU()
        )
        self.sts_classifier3 = nn.Sequential( self.common,
            nn.Linear(BERT_HIDDEN_SIZE, 1, bias = True), nn.ReLU()
        )
        self.sts_classifier_head = nn.Sequential(
            nn.Linear(3, 1, bias = True)
        )

        # Learning to learn. Add additional head to learn the task to be performed 
        # Add soft labels for each dataset for each tasks if needed and train on all the datasets 
        # Pretraining: SimCSE and then MLM loss




    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).

        output = self.bert(input_ids, attention_mask)
        pooler_output = self.dropout(output["pooler_output"])
        return pooler_output

    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        hidden = self.forward(input_ids, attention_mask)
        logits = self.sst_classifier(hidden)
        return logits


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''

        # hidden_1 = self.forward(input_ids_1, attention_mask_1)
        # hidden_2 = self.forward(input_ids_2, attention_mask_2)
        
        input_id1 = torch.concat((input_ids_1, input_ids_2), dim = 1)
        attention1 = torch.concat((attention_mask_1,attention_mask_2), dim = 1)
        hidden1 = self.forward(input_id1, attention1)


        input_id2 = torch.concat((input_ids_2, input_ids_1), dim = 1)
        attention2 = torch.concat((attention_mask_2, attention_mask_1), dim = 1)
        hidden2 = self.forward(input_id2, attention2)
        
        hidden3 = torch.sub(hidden1, hidden2)

        l1= self.paraphrase_classifier1(hidden1)
        l2= self.paraphrase_classifier2(hidden2)
        l3= self.paraphrase_classifier3(hidden3)

        # we use dim = 1 because we want the embeddings to be concatenated sequentially
        # hidden = torch.cat((hidden_1, hidden_2), dim=1)
        
        logits = self.paraphrase_classifier_head(torch.concat((l1,l2,l3), dim = 1 ))
        
        return logits


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation, and handled as a logit by the appropriate loss function.
        '''
        # inp = torch.concat((input_ids_1, tokenizer._sep_token, input_ids_2), dim = 1)
        # attn = torch.concat((attention_mask_1, attention_mask_2), dim = 1)
        # print('shape')
        # print(inp.shape, attn.shape)
        # hidden = self.forward(inp, attn)
        # print(input_ids_1)
        # print(input_ids_2)

        # print('attention masks ')
        # print(attention_mask_1)
        # print(attention_mask_2)
        
        # print(self.tokenizer._sep_token)
        # print(self.tokenizer.sep_token_id)    
        # print(input_ids_1.shape, input_ids_2.shape)
           
        input_id1 = torch.concat((input_ids_1, input_ids_2), dim = 1)
        attention1 = torch.concat((attention_mask_1,attention_mask_2), dim = 1)
        hidden1 = self.forward(input_id1, attention1)


        input_id2 = torch.concat((input_ids_2, input_ids_1), dim = 1)
        attention2 = torch.concat((attention_mask_2,attention_mask_1), dim = 1)
        hidden2 = self.forward(input_id2, attention2)
        
        hidden3 = torch.sub(hidden1, hidden2)
        
        l1= self.sts_classifier1(hidden1)
        l2= self.sts_classifier2(hidden2)
        l3= self.sts_classifier3(hidden3)

        # input_id= self.tokenizer.build_inputs_with_special_tokens(input_ids_1,input_ids_2)
        # attention = self.tokenizer.get_special_tokens_mask(input_ids_1,input_ids_2)

        # print('Pair rep')
        # print(input_id.shape)
        # print(attention)

        # hidden_1 = self.forward(input_ids_1, attention_mask_1)
        # hidden_2 = self.forward(input_ids_2, attention_mask_2)        
        # sub = hidden_1 - hidden_2
        # cos = torch.nn.CosineSimilarity(dim =1)(hidden_1,hidden_2).reshape(-1,1)
        # hidden = torch.cat((hidden_1, hidden_2, sub, cos), dim=1)
        # hidden = torch.cat((hidden_1, hidden_2, sub), dim=1)

        # hidden = self.forward(input_id,attention)
        # print(hidden.shape)

        logits = self.sts_classifier_head(torch.concat((l1,l2,l3), dim =1 ))
        
        return logits


def save_model(model, optimizer, args, config, filepath):
    
    
    if args.grad_surgery ==True: 
        optimizer = optimizer.optimizer
    
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")



# Perform model evaluation in terms by averaging accuracies across tasks.
def model_eval_multitask_dataset(sentiment_dataloader,
                         paraphrase_dataloader,
                         sts_dataloader,
                         model, device, dataset="all"):
    model.eval()  # switch to eval model, will turn off randomness like dropout

    with torch.no_grad():
        para_y_true = []
        para_y_pred = []
        para_sent_ids = []

        # Evaluate paraphrase detection.
        if dataset == "all" or dataset == "paraphrase":
            for step, batch in enumerate(tqdm(paraphrase_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
                (b_ids1, b_mask1,
                b_ids2, b_mask2,
                b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                            s2_featurebatch['token_ids_2'], batch['attention_mask_2'],
                            batch['labels'], batch['sent_ids'])

                b_ids1 = b_ids1.to(device)
                b_mask1 = b_mask1.to(device)
                b_ids2 = b_ids2.to(device)
                b_mask2 = b_mask2.to(device)

                logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
                y_hat = logits.sigmoid().round().flatten().cpu().numpy()
                b_labels = b_labels.flatten().cpu().numpy()

                para_y_pred.extend(y_hat)
                para_y_true.extend(b_labels)
                para_sent_ids.extend(b_sent_ids)

            paraphrase_accuracy = np.mean(np.array(para_y_pred) == np.array(para_y_true))
        else:
            paraphrase_accuracy = 0.0

        sts_y_true = []
        sts_y_pred = []
        sts_sent_ids = []


        # Evaluate semantic textual similarity.
        if dataset == "all" or dataset == "sts":
            for step, batch in enumerate(tqdm(sts_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
                (b_ids1, b_mask1,
                b_ids2, b_mask2,
                b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                            batch['token_ids_2'], batch['attention_mask_2'],
                            batch['labels'], batch['sent_ids'])

                b_ids1 = b_ids1.to(device)
                b_mask1 = b_mask1.to(device)
                b_ids2 = b_ids2.to(device)
                b_mask2 = b_mask2.to(device)

                logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
                y_hat = logits.flatten().cpu().numpy()
                b_labels = b_labels.flatten().cpu().numpy()

                sts_y_pred.extend(y_hat)
                sts_y_true.extend(b_labels)
                sts_sent_ids.extend(b_sent_ids)
            pearson_mat = np.corrcoef(sts_y_pred,sts_y_true)
            sts_corr = pearson_mat[1][0]
        else:
            sts_corr = 0.0


        sst_y_true = []
        sst_y_pred = []
        sst_sent_ids = []

        # Evaluate sentiment classification.
        if dataset == "all" or dataset == "sst":
            for step, batch in enumerate(tqdm(sentiment_dataloader, desc=f'eval', disable=TQDM_DISABLE)):
                b_ids, b_mask, b_labels, b_sent_ids = batch['token_ids'], batch['attention_mask'], batch['labels'], batch['sent_ids']

                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)

                logits = model.predict_sentiment(b_ids, b_mask)
                y_hat = logits.argmax(dim=-1).flatten().cpu().numpy()
                b_labels = b_labels.flatten().cpu().numpy()

                sst_y_pred.extend(y_hat)
                sst_y_true.extend(b_labels)
                sst_sent_ids.extend(b_sent_ids)

            sentiment_accuracy = np.mean(np.array(sst_y_pred) == np.array(sst_y_true))
        else:
            sentiment_accuracy = 0.0

        print(f'Paraphrase detection accuracy: {paraphrase_accuracy:.3f}')
        print(f'Sentiment classification accuracy: {sentiment_accuracy:.3f}')
        print(f'Semantic Textual Similarity correlation: {sts_corr:.3f}')

        return (paraphrase_accuracy, para_y_pred, para_sent_ids,
                sentiment_accuracy,sst_y_pred, sst_sent_ids,
                sts_corr, sts_y_pred, sts_sent_ids)


## Currently only trains on sst dataset
def train_multitask(args):
    #Acc Metric
    global best_dev
    
    # Load data
    # Create the data and its corresponding datasets and dataloader
    sst_train_data_raw, num_labels,para_train_data_raw, sts_train_data_raw = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data_raw, num_labels,para_dev_data_raw, sts_dev_data_raw = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data_raw, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data_raw, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size= args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
                                      
                                      
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    
    para_train_data = SentencePairDataset(para_train_data_raw, args)
    para_dev_data = SentencePairDataset(para_dev_data_raw, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=para_dev_data.collate_fn)

    sts_train_data = SentencePairDataset(sts_train_data_raw, args, isRegression=True)
    sts_dev_data = SentencePairDataset(sts_dev_data_raw, args, isRegression=True)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)

    total_batches = int(len(para_train_dataloader)/5)

    def dev_metrics():
        global best_dev
        print(best_dev)
        dev_sst, dev_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)
        dev_para = model_eval_para(para_dev_dataloader, model, device)
        dev_corr = model_eval_sts(sts_dev_dataloader, model, device)
        average = (dev_sst + dev_para + dev_corr)/3
        
        print(f"dev_sst :: {dev_sst :.3f}, dev_para :: {dev_para :.3f}, dev_corr :: {dev_corr :.3f}, average :: {average :.3f}")
        return  dev_sst, dev_para, dev_corr, average


    # Implementation of Gradient Surgery with MT-DNN approach 
    def mt_dnn2():
        global best_dev
        # sts_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
        #                                 collate_fn=sst_train_data.collate_fn)
        # sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
        #                               collate_fn=sst_train_data.collate_fn)
        # para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
        #                             collate_fn=para_train_data.collate_fn)       

        DATA_SIZE = args.batch_size
        sst_iter = iter(sst_train_dataloader)
        sts_iter = iter(sts_train_dataloader)
        para_iter= iter(para_train_dataloader)
            
        for epoch in tqdm(range(args.epochs)):

            is_save_model = True
            print(f"Executing epoch {epoch + 1} out of {args.epochs} epochs")

            train_loss = 0
            train_acc_sst = 0
            train_pearson_sts= 0
            train_acc_para =0
            
            
            for nbatch in tqdm( range(total_batches)):
                


                choice1 = random.randint(0,1)*random.randint(0,1)
                choice2 = random.randint(0,1)
                choice3 = random.randint(0,1)*random.randint(0,1)

                choice1=choice2=choice3=1

                model.train()
                optimizer.zero_grad()
                
                loss_sst =loss_para = loss_sts = torch.zeros((0),requires_grad = True, device= torch.device('cuda'))
                
                if choice1 == 1:
                ## SST Gradient 
                    try:
                        batch = next(sst_iter)
                    except StopIteration:
                        
                        sst_iter = iter(DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                            collate_fn=sst_train_data.collate_fn))
                        batch = next(sst_iter)

                    b_ids, b_mask, b_labels = (batch['token_ids'], 
                                                batch['attention_mask'], 
                                                batch['labels'])
                    
                    b_ids = b_ids.to(device)
                    b_mask = b_mask.to(device)
                    b_labels = b_labels.to(device)
                    
                    logits = model.predict_sentiment(b_ids, b_mask)
                    loss_sst = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size
                    
                    # Paraphrase part

                if choice2 == 1:
                    try:
                        batch = next(para_iter)
                    except StopIteration:
                        # para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                        #                     collate_fn=para_train_data.collate_fn)
                        para_iter = iter( DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                            collate_fn=para_train_data.collate_fn))
                        batch = next(para_iter)


                    b_ids1, b_mask1,b_ids2, b_mask2, b_labels, b_sent_ids = (batch['token_ids_1'], 
                                                                                batch['attention_mask_1'],
                                                                                batch['token_ids_2'], 
                                                                                batch['attention_mask_2'],
                                                                                batch['labels'], 
                                                                                batch['sent_ids']
                                                                                )

                    b_ids1 = b_ids1.to(device)
                    b_mask1 = b_mask1.to(device)
                    b_ids2 = b_ids2.to(device)
                    b_mask2 = b_mask2.to(device)

                    # optimizer.zero_grad()
                    logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
                    
                    y_hat = logits.sigmoid().round().flatten().cpu().detach().numpy()

                    b_labels = b_labels.type(torch.FloatTensor).to(device)
                    loss_para = F.binary_cross_entropy_with_logits(logits.flatten(), b_labels, reduction='sum') / args.batch_size

                ## Loss Similarity
                if choice3==1:
                    try:
                        batch = next(sts_iter)
                    except StopIteration:
                        sts_iter = iter(DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                            collate_fn=sts_train_data.collate_fn))
                        batch = next(sts_iter)


                    b_ids1, b_mask1, b_ids2, b_mask2, b_labels, b_sent_ids = (batch['token_ids_1'], 
                                                                                batch['attention_mask_1'],
                                                                                batch['token_ids_2'], 
                                                                                batch['attention_mask_2'],
                                                                                batch['labels'], 
                                                                                batch['sent_ids'])

                    b_ids1 = b_ids1.to(device)
                    b_mask1 = b_mask1.to(device)
                    b_ids2 = b_ids2.to(device)
                    b_mask2 = b_mask2.to(device)

                    # optimizer.zero_grad()
                    logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
                    y_hat = logits.flatten().cpu().detach().numpy()
                    
                    b_labels = b_labels.flatten().type(torch.FloatTensor).to(device)
                    loss_sts  = F.mse_loss(logits.flatten(), b_labels) 
                # Gradient computation

                

                loss = [loss_sst, 5*loss_sts, loss_para]
                
                
                if args.grad_surgery == True:
                    optimizer.pc_backward(loss)
                else:
                    loss= sum(loss)
                    loss.backward()

                if args.grad_clipping ==True:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                
                optimizer.step()

                train_loss += loss_para.item() + loss_sst.item() + loss_sts.item()

                torch.cuda.empty_cache()
          


                if(nbatch%250 == 0):
                    a, train_f1, *_ = model_eval_sst(sst_train_dataloader, model, device)
                    train_acc_sst=a

                    a = model_eval_sts(sts_train_dataloader,model, device)
                    train_pearson_sts=a

                    # a, train_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)
                    # dev_acc_sst=a

                    # a = model_eval_sts(sts_dev_dataloader,model, device)
                    # dev_pearson_sts=a
                    # a = model_eval_para(para_train_dataloader, model, device)
                    # train_acc_para+=a    

                    print(f'Training Loss {train_loss/(nbatch+1)} SST accuracy {train_acc_sst} STS pearson {train_pearson_sts}' )
                    # print(f'SST dev accuracy {dev_acc_sst} STS dev pearson {dev_pearson_sts}')

                    if is_save_model:
                        print('Running dev metrics')
                        dev_sst, dev_para, dev_corr, average = dev_metrics()

                        if average >= best_dev:
                            best_dev = average
                            save_model(model, optimizer, args, config, args.filepath)
                            print('new_aggregate',average)
                            print('best_aggregate',best_dev)
                        else:
                            # saved = torch.load(args.filepath)
                            # config = saved['model_config']

                            # model = MultitaskBERT(config)
                            # model.load_state_dict(saved['model'])
                            # model = model.to(device)
                            print(f'Model does not perform better than the previous one New average {average} previous average {best_dev}')

                            # print(f"Loaded model from checkpoint {args.filepath}")

    # Evaluation of SNLI dataset during pretraining the BERT model using Supervised Constrastive Learning
    def evaluate_SNLI(model,args, dataloader):
        

        sim = Similarity(0.05)
        y_true_list =[]
        y_pred_list =[]
        for index, batch in enumerate(dataloader):

            (anchor_token_ids, anchor_attention_mask , anchor_token_type_ids,
            positve_token_ids, positve_attention_mask , positve_token_type_ids,
            negative_token_ids, negative_attention_mask , negative_token_type_ids,
            ) = batch

            anchor_embedding = model(anchor_token_ids.to(device),anchor_attention_mask.to(device))
            positive_embedding = model(positve_token_ids.to(device),positve_attention_mask.to(device))
            negative_embedding = model(negative_token_ids.to(device),negative_attention_mask.to(device))
            
            anchor_embedding = anchor_embedding/ torch.linalg.norm(anchor_embedding, dim = 0) 
            positive_embedding = positive_embedding/ torch.linalg.norm(positive_embedding,dim = 0)
            negative_embedding = negative_embedding/ torch.linalg.norm(negative_embedding,dim = 0)
                

            sim_pos = sim(anchor_embedding,positive_embedding)
            sim_pos = sim_pos/ torch.linalg.norm(sim_pos, dim = 0) 
            sim_neg = sim(anchor_embedding,positive_embedding)
            sim_neg = sim_neg/ torch.linalg.norm(sim_neg, dim = 0) 


            y_pred = torch.cat([sim_pos, sim_neg]).detach().cpu().numpy()
            y_true = torch.cat([torch.ones((args.batch_size)), torch.zeros((args.batch_size))]).detach().cpu().numpy()
            
            y_true_list.extend( y_true>0.5 )
            y_pred_list.extend(y_pred)
        
        acc = np.mean( (np.array(y_true_list))== np.array(y_pred_list))
        print(f"Dev Accuracy is {acc}")

    # SimCSE Pretraining Function 
    def simcse(args):
        
        dev_best=0
        
        snli_train_dataset = load_snli(split= 'train')
        snli_train_dataset = SNLI(snli_train_dataset, args)
        snli_train_dataloader = DataLoader(snli_train_dataset, shuffle=True, batch_size= args.batch_size,
                                            collate_fn=snli_train_dataset.pad_data)

        snli_dev_dataset = load_snli(split= 'dev')
        snli_dev_dataset= SNLI(snli_dev_dataset, args)
        snli_dev_dataloader = DataLoader(snli_dev_dataset, shuffle=True, batch_size= args.batch_size,
                                            collate_fn=snli_dev_dataset.pad_data)


        temperature_loss = TemperatureLoss(temperature=0.05)
        loss_fn = torch.nn.TripletMarginLoss()
        sim = Similarity(0.05)
        optimizer = torch.optim.AdamW(model.parameters(), lr= args.lr)

        for epoch in range(args.epochs):

            train_loss = 0
            pos_simtot=0 
            neg_simtot=0
            for index, batch in enumerate(snli_train_dataloader):
                # print(batch)

            
                (anchor_token_ids, anchor_attention_mask , anchor_token_type_ids,
                positve_token_ids, positve_attention_mask , positve_token_type_ids,
                negative_token_ids, negative_attention_mask , negative_token_type_ids,
                ) = batch

                anchor_embedding = model(anchor_token_ids.to(device),anchor_attention_mask.to(device))
                positive_embedding = model(positve_token_ids.to(device),positve_attention_mask.to(device))
                negative_embedding = model(negative_token_ids.to(device),negative_attention_mask.to(device))

                anchor_embedding = anchor_embedding/ torch.linalg.norm(anchor_embedding, dim = 0) 
                positive_embedding = positive_embedding/ torch.linalg.norm(positive_embedding,dim = 0)
                negative_embedding = negative_embedding/ torch.linalg.norm(negative_embedding,dim = 0)
                



                loss = loss_fn(anchor_embedding,positive_embedding,negative_embedding)
                # loss = temperature_loss(anchor_embedding.to(device), positive_embedding.to(device), negative_embedding.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                print(f"Training Loss {train_loss/(index+1)}")

                sim_pos = sim(anchor_embedding,positive_embedding)
                sim_pos = sim_pos/ torch.linalg.norm(sim_pos, dim = 0) 
                sim_neg = sim(anchor_embedding,positive_embedding)
                sim_neg = sim_neg/ torch.linalg.norm(sim_neg, dim = 0) 

                # print('Shape of similarity', sim(anchor_embedding, positive_embedding).shape)
                pos_simtot += torch.mean((sim_pos))
                neg_simtot += torch.mean((sim_neg))

                # print(sim(anchor_embedding, positive_embedding))
                print(f"Cosine Similarity  for positive pairs is {pos_simtot/(index+1)} and negative pairs is {neg_simtot/(index+1)}" )

                if index%250 ==0:
                    evaluate_SNLI(model,args,snli_dev_dataloader)
                    
                    save_model(model,optimizer,args,config, filepath=args.filepath)
                


    def mt_dnn():
        # ~Each epoch cycle through the random dataloader.~
        # Batch with random task specific data points  (Changed dataloader ) 
        # Calculate the losses using the identifier and add the gradients  
        # Handle losses using parameters 1/3 weight 
        # clip gradient and 
        # Check if detach is a bottleneck 
        # Fix batch size and the learning rate 
        import random
        
        NUM_CHOICES = 3
        DATA_SIZE = 2048
        
        for epoch in range(args.epochs*10):
            is_save_model = True if (((1 + epoch) % 5) == 0) else False
            print(f"Executing epoch {epoch + 1} out of {args.epochs} epochs...")
        
            choice = random.randrange(1 + NUM_CHOICES)
            choice = 4
            # sampler = torch.utils.data.RandomSampler(
            #     data_source=sst_train_dataloader,
            #     num_samples=args.batch_size
            # )
            # data_loader = DataLoader(sst_train_dataloader, sampler=sampler, batch_size=args.batch_size)
            
            if choice == 0:
                print("\tTraining on sentiment...")
                sst_train_data_raw_sample = random.choices(sst_train_data_raw, k=DATA_SIZE)
                
                sst_train_data_sample = SentenceClassificationDataset(sst_train_data_raw_sample, args)
                
                sst_train_dataloader_sample = DataLoader(sst_train_data_sample, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=sst_train_data_sample.collate_fn)
                
                train_sentiment(sst_train_dataloader_sample, is_save_model)
            
            elif choice == 1 or choice == 2:
                print("\tTraining on paraphrase...")
                para_train_data_raw_sample = random.choices(para_train_data_raw, k=DATA_SIZE)
                
                para_train_data_sample = SentencePairDataset(para_train_data_raw_sample, args)
                
                para_train_dataloader_sample = DataLoader(para_train_data_sample, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=para_train_data_sample.collate_fn)

                train_paraphrase(para_train_dataloader_sample, is_save_model)
            else:
                print("\tTraining on similarity...")
                sts_train_data_raw_sample = random.choices(sts_train_data_raw, k=DATA_SIZE)
                
                sts_train_data_sample = SentencePairDataset(sts_train_data_raw_sample, args, isRegression=True)
                
                sts_train_dataloader_sample = DataLoader(sts_train_data_sample, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=sts_train_data_sample.collate_fn)
                
                train_similarity(sts_train_dataloader_sample, is_save_model)

    def train_sentiment(sst_train_dataloader=sst_train_dataloader, is_save_model=True):
        best_dev_acc = 0
        global best_dev
        # Run for the specified number of epochs
        for epoch in range(args.epochs):
            
            if epoch ==1: return
            model.train()
            train_loss = 0
            num_batches = 0
            
            data = tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)
            for step, batch in enumerate(data):
                b_ids, b_mask, b_labels = (batch['token_ids'],
                                        batch['attention_mask'], batch['labels'])

                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                b_labels = b_labels.to(device)

                optimizer.zero_grad()
                logits = model.predict_sentiment(b_ids, b_mask)
                
                loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size
                
                
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1
                # break
                # print(f"Epoch: {epoch}, batch: {step + 1} / {len(data)}")

            train_loss = train_loss / (num_batches)

            train_acc, train_f1, *_ = model_eval_sst(sst_train_dataloader, model, device)
            # dev_sst, dev_f1, *_ = model_eval_sst(sst_dev_dataloader, model, device)
            
            if is_save_model:
                print('Running dev metrics')
                dev_sst, dev_para, dev_corr, average = dev_metrics()


                if average >= best_dev:
                    best_dev = average
                    save_model(model, optimizer, args, config, args.filepath)
                print('new_aggregate',average)
                print('best_aggregate',best_dev)
                print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train_acc :: {train_acc :.3f} dev_sst :: {dev_sst :.3f}")
            else:
                print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train_acc :: {train_acc :.3f}")

            # if dev_acc > best_dev_acc:
            #     best_dev_acc = dev_acc
            #     save_model(model, optimizer, args, config, args.filepath)

    
    #Acc Metric
    def train_paraphrase(para_train_dataloader=para_train_dataloader, is_save_model=True):
        # best_dev_acc = 0
        global best_dev
        # Run for the specified number of epochs
        for epoch in range(args.epochs):
            
            if epoch == 1: return

            model.train()
            train_loss = 0
            num_batches = 0
            
            para_y_true = []
            para_y_pred = []
            para_sent_ids = []
            
            data = tqdm(para_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)
            for step, batch in enumerate(data):
                (b_ids1, b_mask1,
                b_ids2, b_mask2,
                b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                            batch['token_ids_2'], batch['attention_mask_2'],
                            batch['labels'], batch['sent_ids'])

                b_ids1 = b_ids1.to(device)
                b_mask1 = b_mask1.to(device)
                b_ids2 = b_ids2.to(device)
                b_mask2 = b_mask2.to(device)

                optimizer.zero_grad()
                logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
                
                y_hat = logits.sigmoid().round().flatten().cpu().detach().numpy()
                
                
                para_y_pred.extend(y_hat)
                para_y_true.extend(b_labels)
                para_sent_ids.extend(b_sent_ids)
                
                # print(b_labels)
                b_labels = b_labels.type(torch.FloatTensor).to(device)
                # s1_feature = model.forward(b_ids1, b_mask1)
                # s2_feature = model.forward(b_ids2, b_mask2)
                # loss = train_loss_function(s1_feature, s2_feature)
                loss = F.binary_cross_entropy_with_logits(logits.flatten(), b_labels, reduction='sum') / args.batch_size
                loss.backward()

                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()
                train_loss += loss.item()
                num_batches += 1
                if(step ==500): break
                # print(f"Epoch: {epoch}, batch: {step + 1} / {len(data)}")
                
            train_loss = train_loss / (num_batches)
            
            train_acc = np.mean(np.array(para_y_pred) == np.array(para_y_true))
            # dev_acc = model_eval_para(para_dev_dataloader, model, device)


            # train_acc= model_eval_para(para_train_dataloader, model, device)
            # dev_acc = model_eval_para(para_dev_dataloader, model, device)
            # print(train_acc,dev_acc)
            
            if is_save_model:
                print('Running dev metrics')
                dev_sst, dev_para, dev_corr, average = dev_metrics()

                if average >= best_dev:
                    best_dev = average
                    save_model(model, optimizer, args, config, args.filepath)
                print('new_aggregate',average)
                
                print('best_aggregate',best_dev)
                print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train_acc :: {train_acc :.3f} dev_para :: {dev_para :.3f}")
            else:
                print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train_acc :: {train_acc :.3f}")

            # # TODO: uses train accuracy and not dev accuracy
            # if dev_acc > best_dev_acc:
            #     best_dev_acc = dev_acc
            #     save_model(model, optimizer, args, config, args.filepath)

    
    # Pearson Coefficient
    def train_similarity(sts_train_dataloader=sts_train_dataloader, is_save_model=True):
        best_dev_sts = 0
        global best_dev

        # Run for the specified number of epochs
        for epoch in range(args.epochs):
            if epoch == 1: return
            model.train()
            train_loss = 0
            num_batches = 0
            
            sts_y_true = []
            sts_y_pred = []
            sts_sent_ids = []
            
            data = tqdm(sts_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE)
            print("Data: ", data)
            for step, batch in enumerate(data):
                (b_ids1, b_mask1,
                b_ids2, b_mask2,
                b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                            batch['token_ids_2'], batch['attention_mask_2'],
                            batch['labels'], batch['sent_ids'])

                b_ids1 = b_ids1.to(device)
                b_mask1 = b_mask1.to(device)
                b_ids2 = b_ids2.to(device)
                b_mask2 = b_mask2.to(device)

                optimizer.zero_grad()
                logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
                y_hat = logits.flatten().cpu().detach().numpy()
                # y_hat = logits.cpu().detach().numpy()
                
                # print(logits.flatten().shape ,b_labels.shape)
                # preds = np.argmax(y_hat, axis = 1)

                # print(y_hat)
                # print(b_labels)
                # sts_y_pred.extend(preds)

                sts_y_pred.extend(y_hat)
                sts_y_true.extend(b_labels)
                sts_sent_ids.extend(b_sent_ids)
                
                b_labels = b_labels.flatten().type(torch.FloatTensor).to(device)
                loss = F.mse_loss(logits.flatten(), b_labels) 
                # loss = F.cross_entropy(logits, b_labels) 
                
                # print(loss.detach())
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

                optimizer.step()

                train_loss += loss.item()
                num_batches += 1
                
                # if(step == 10): break
                # print(f"Epoch: {epoch}, batch: {step + 1} / {len(data)}")

            train_loss = train_loss / (num_batches)
            
            pearson_mat = np.corrcoef(sts_y_pred, sts_y_true)
            train_corr = pearson_mat[1][0]
            # dev_corr = model_eval_sts(sts_dev_dataloader, model, device)
            
            # print('Training Corr', train_corr)
            if is_save_model:
                print('Running dev metrics')
                dev_sst, dev_para, dev_corr, average = dev_metrics()


                if average >= best_dev:
                    best_dev = average
                    save_model(model, optimizer, args, config, args.filepath)
                print('new_aggregate',average)
                print('best_aggregate',best_dev)
                print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train_corr :: {train_corr :.3f} dev_corr :: {dev_corr :.3f}")
            else:
                print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train_corr :: {train_corr :.3f}")
            # if dev_corr > best_dev_sts:
            #     best_dev_sts = dev_corr
            #     save_model(model, optimizer, args, config, args.filepath)

            

            # train_sts= model_eval_sts(sts_train_dataloader, model, device)
            # print(train_sts,dev_sts)

    best_dev = 0

    # Init model
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    

    print('modelname exits?')
    print(os.path.isfile(args.checkpoint))
    print(args.checkpoint)

    model = MultitaskBERT(config)
    model = model.to(device)
 

    if args.checkpoint and os.path.isfile(args.checkpoint):

        saved = torch.load(args.checkpoint)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print('Loaded Model from checkpoint successfully!')
    
    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100000, eta_min= 0.000001)
    
    if args.grad_surgery == True:
        optimizer = PCGrad(optimizer, 'sum')

    # Training Method

    if args.method == 'simcse': 
        # pass
        simcse(args)
    

    elif args.method =='sst':

        for epoch in range(args.epochs):
            train_sentiment()


    elif args.method =='sts':

        for epoch in range(args.epochs):
            train_similarity(is_save_model=False)

    
    elif args.method =='pd':
        
        for epoch in range(args.epochs):
            train_paraphrase()


    elif args.method =='roundrobin':

        for epoch in range(args.epochs):
            pass
            # train_sentiment()
            # train_similarity()
            # train_paraphrase()
            # train_sentiment()
            # train_similarity


    elif args.method =='mt-dnn':
        mt_dnn()
    
    elif args.method =='mt-dnn2':
        # pass
        # train_paraphrase()
        # train_similarity()
        # train_similarity()
        # train_paraphrase()
        # train_sentiment()
        # train_sentiment()
        # train_paraphrase()
        mt_dnn2()
       
    
    # mt_dnn()
    
        


        


def test_model(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        test_model_multitask(args, model, device)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    # hyper parameters
    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-5)
    
    # New arguments
    parser.add_argument("--simcse", help = ' Implementation of Constrastive Learning', type = str, default=None)
    parser.add_argument("--modelname", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--grad_surgery", action= 'store_true')
    parser.add_argument("--method", help = 'choose from mt-dnn mt-dnn2 or roundrobin', default='roundrobin')
    parser.add_argument("--grad_clipping", action= 'store_true')



    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
        

    args.filepath = f'{args.modelname}-{args.option}-{args.epochs}-{args.lr}-{args.batch_size}-{args.hidden_dropout_prob}-multitask.pt' # save path
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_multitask(args)
    test_model(args)


    # pretrained bert weights (frozen)   --> Linear Layer (Trainable)   : Pretrain
    # larger learning rate



    # pretrained bert weights (Trainable)   --> Linear Layer (Trainable)   : Finetune

    # pretrained bert weights (Trainable) --> simcse --Learns the embeddings 
    
    # SimCSE using SNLI dataset --> finetune bert weights  
    # Negative ranking loss --> finetune bert weights 
    
    # Negative and SimCSE dataloader for SNLI 