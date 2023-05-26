from __future__ import unicode_literals, print_function, division
from io import open
from torch.utils.data import DataLoader,TensorDataset
from torchvision import models,datasets
import unicodedata
import string
import re
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch.optim as optim
import os
import glob
import time
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import math
from types import SimpleNamespace
import itertools
import argparse
import wandb

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
device = torch.device('mps')
PAD_token = -1  # Used for padding short sentences
SOS_token = 0  # Start-of-sentence token
EOS_token = 1  # End-of-sentence token
MAX_LENGTH = 24    

default_config = SimpleNamespace(
	wandb_project = 'rnn',
	wandb_entity = 'uttakarsh05',
	n_iterations = 4000, 
    input_lang = 'eng',
    output_lang = 'hin',
	encoder_optimizer = 'adam',
    decoder_optimizer = 'adam',
	encoder_learning_rate = 1e-3,
    decoder_learning_rate=1e-3,
	momentum = 0.9,
	beta = 0.9,
	beta1 = 0.9,
	beta2 = 0.99,
	weight_decay = 0,
	weight_initialization = 'He',
    clip = 50,
    teacher_forcing_ratio = 0.5,
	encoder_embedding_size = 16,
	decoder_embedding_size = 32,
    n_layers = 2,
    cell_type = 'lstm',
    bidirectional = 'False',
    batch_size = 32,
    hidden_size = 128,
	dropout = 0.2,
    print_every = 50,
	activation = 'relu',
    layer_norm = 'True',
    data_dir = 'data',
    device = 'mps'
)

def parse_args():
	argparser = argparse.ArgumentParser(description = 'Processing Hyperparameters')
	argparser.add_argument('-wp','--wandb_project',type = str,default = default_config.wandb_project,help = 'wandb project name')
	argparser.add_argument('-we','--wandb_entity',type = str,default = default_config.wandb_entity,help = 'wandb username/entity name')
	argparser.add_argument('-n_i','--n_iterations',type = int,default = default_config.n_iterations,help = 'no of iterations for training')
	argparser.add_argument('-ip_lang','--input_lang',type = str,default = default_config.input_lang,help = 'name of the input language (default is english)')
	argparser.add_argument('-op_lang','--output_lang',type = str,default = default_config.output_lang,help = 'name of the output language')
	argparser.add_argument('-e_o','--encoder_optimizer',type = str,default = default_config.encoder_optimizer,help = 'name of encoder optimizer')
	argparser.add_argument('-d_o','--decoder_optimizer',type = str,default = default_config.decoder_optimizer,help = 'name of decoder optimizer')
	argparser.add_argument('-e_lr','--encoder_learning_rate',type = float,default = default_config.encoder_learning_rate,help = 'learning rate of the encoder')
	argparser.add_argument('-d_lr','--decoder_learning_rate',type = float,default = default_config.decoder_learning_rate,help = 'learning rate of the decoder')
	argparser.add_argument('-m','--momentum',type = float,default = default_config.momentum,help = 'beta value used for momentum optimizer')
	argparser.add_argument('-beta','--beta',type = float,default = default_config.beta,help = 'beta value used for rmsprop')
	argparser.add_argument('-beta1','--beta1',type = float,default = default_config.beta1,help = 'beta1 used by adam and nadam')
	argparser.add_argument('-beta2','--beta2',type = float,default = default_config.beta2,help = 'beta2 used by adam and nadam')
	argparser.add_argument('-w_d','--weight_decay',type = float,default = default_config.weight_decay,help = 'weight decay (lamda) value for l2 regularization')
	argparser.add_argument('-w_i','--weight_init',type = str,default = default_config.weight_initialization,help = 'activation name')
	argparser.add_argument('-cl','--clip',type = float,default = default_config.clip,help = 'Threshold for gradient clipping')
	argparser.add_argument('-t_f','--teacher_forcing_ratio',type = float,default = default_config.teacher_forcing_ratio,help = 'Defines the ratio of training done using teacher forcing ')
	argparser.add_argument('-e_emb','--encoder_embedding_size',type = int, default = default_config.encoder_embedding_size , help = 'resolution of input image')
	argparser.add_argument('-d_emb','--decoder_embedding_size',type = int, default = default_config.decoder_embedding_size , help = 'no of channels in the input image')
	argparser.add_argument('-n_l','--n_layers',type = int, default = default_config.n_layers , help = 'no of layers in the encoder and decoder')
	#argparser.add_argument('-d_n','--decoder_n_layers',type = int, default = default_config.decoder_n_layers , help = 'no of layers in the decoder')
	argparser.add_argument('-c','--cell_type',type = str,default = default_config.cell_type,help = 'cell type in encoder and decoder')
	argparser.add_argument('-bi','--bidirectional',type = str,default = default_config.bidirectional,help = 'if the cell type is bidirectional or not')
	argparser.add_argument('-b','--batch_size',type = int,default = default_config.batch_size,help = 'batch size for training one iterations')
	argparser.add_argument('-h_s','--hidden_size',type = int,default = default_config.hidden_size,help = 'hidden state size in the encoder and decoder layer')
	argparser.add_argument('-d','--dropout',type = float,default = default_config.dropout,help = 'probability of a neuron to be dropped')
	argparser.add_argument('-p_e','--print_every',type = int,default = default_config.print_every,help = 'no of iterations for calculating training loss')
	argparser.add_argument('-a','--activation',type = str,default = default_config.activation,help = 'activation name')
	argparser.add_argument('-l_n','--layer_norm',type=str , default=default_config.layer_norm, help='layer normalization ')
	argparser.add_argument('-d_dir','--data_dur',type=str , default=default_config.data_dir, help='path to the data folder')
	argparser.add_argument('-dev','--device',type=str , default=default_config.device, help='device used for training the model') 
	args = argparser.parse_args()
	vars(default_config).update(vars(args))
	return 


class Lang:
    def __init__(self, name):
        self.name = name
        self.letter2index = {}
        #self.word2count = {}
        self.index2letter = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addWord(self, name):
        for l in name:
            self.addLetter(l)

    def addLetter(self, word):
        if word not in self.letter2index:
            self.letter2index[word] = self.n_words
            #self.word2count[word] = 1
            self.index2letter[self.n_words] = word
            self.n_words += 1
        #else:
        #    self.word2count[word] += 1
        

def readLangs(lang1,lang2):
    
    lines = open('data/%s/%s_train.csv' % (lang2, lang2), encoding='utf-8').read().strip().split('\n')
    pairs = [[word for word in l.split(',')] for l in lines]
    val_lines = open('data/%s/%s_valid.csv' % (lang2, lang2), encoding='utf-8').read().strip().split('\n')
    val_pairs = [[word for word in l.split(',')] for l in val_lines]
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    
    return input_lang,output_lang,pairs,val_pairs


def prepareData(lang1,lang2):
    input_lang, output_lang, pairs,val_pairs = readLangs(lang1, lang2)
    print("Read %s word pairs" % len(pairs))
    for pair in pairs:
        input_lang.addWord(pair[0])
        output_lang.addWord(pair[1])
        
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    
    return input_lang, output_lang, pairs,val_pairs

#input_lang, output_lang, pairs = prepareData('eng', 'hin')   

def indexesFromWord(lang,word ):
    l = []
    for letter in word:
        if letter not in lang.letter2index:
            l.append(PAD_token)
        else:
            l.append(lang.letter2index[letter])
    l = l+[EOS_token]
    return l


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# Returns padded input sequence tensor and lengths
def inputVar(l, input_lang):
    indexes_batch = [indexesFromWord(input_lang, word) for word in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, output_lang):
    indexes_batch = [indexesFromWord(output_lang, word) for word in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.BoolTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

# Returns all items for a given batch of pairs
def batch2TrainData( pair_batch,input_lang,output_lang):
    pair_batch.sort(key=lambda x: len(x[0]), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, input_lang)
    output, mask, max_target_len = outputVar(output_batch, output_lang)
    return inp, lengths, output, mask, max_target_len


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def get_optimizer(optimizer,lr,momentum,beta,beta1,beta2,network,weight_decay):
    optimizer = optimizer.lower()
    
    if optimizer=='sgd':
        opt = optim.SGD(network.parameters(),lr = lr ,weight_decay=weight_decay)
    
    elif optimizer=='momentum':
        opt = optim.SGD(network.parameters(),lr = lr,momentum = momentum ,weight_decay=weight_decay)
    
    elif optimizer=='nesterov':
        opt = optim.SGD(network.parameters(),lr = lr , momentum = beta,weight_decay=weight_decay)
    
    elif optimizer == 'adam':
        opt = optim.Adam(network.parameters(),lr = lr , betas = (beta1,beta2),weight_decay=weight_decay)
    
    elif optimizer == 'nadam':
        opt = optim.NAdam(network.parameters(),lr = lr , betas = (beta1,beta2),weight_decay=weight_decay)
    
    elif optimizer == 'rmsprop':
        opt = optim.RMSprop(network.parameters(),lr = lr ,weight_decay=weight_decay)
    
    return opt

def get_activation(activation):
    activation  = activation.lower()
    if activation =='relu':
        g = nn.ReLU()
    
    elif activation == 'tanh':
        g = nn.Tanh()
    
    elif activation == 'silu':
        g = nn.SiLU()
        
    elif activation == 'gelu':
        g = nn.GELU()
    
    elif activation =='celu':
        g = nn.CELU()
        
    elif activation == 'leakyrelu':
        g = nn.LeakyReLU()
    
    elif activation == 'elu':
        g = nn.ELU()
    
    elif activation =='selu':
        g = nn.SELU() 
    
    return g
    

"""def tensorFromWord(lang, word):
    indexes = indexesFromWord(lang, word)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromWord(input_lang, pair[0])
    target_tensor = tensorFromWord(output_lang, pair[1])
    return (input_tensor, target_tensor)
"""

class EncoderRNN(nn.Module):
    def __init__(self,input_size,embedding_size,hidden_size,n_layers = 1,dropout=0,cell_type = 'lstm',bidirectional = False):
        super(EncoderRNN,self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size,embedding_size)
        self.cell_type = cell_type.lower()
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        if self.cell_type=='rnn':
            self.model = nn.RNN(embedding_size,hidden_size,bidirectional = self.bidirectional,num_layers=self.n_layers,dropout=(0 if n_layers == 1 else dropout) )
            
        elif self.cell_type=='gru':
            self.model = nn.GRU(embedding_size,hidden_size,bidirectional = self.bidirectional,num_layers=self.n_layers,dropout=(0 if n_layers == 1 else dropout) )
        
        elif self.cell_type=='lstm':
            self.model = nn.LSTM(embedding_size,hidden_size,bidirectional = self.bidirectional,num_layers=self.n_layers,dropout=(0 if n_layers == 1 else dropout) )
        
    
    def forward(self,input_seq,input_length,hidden=None):
        #print('input length = ',input_length)
        #print('before embedding')
        embedded = self.embedding(input_seq)
        #print('before unpadding')
        packed = nn.utils.rnn.pack_padded_sequence(embedded,input_length)
        #print('before model')
        #print('packed ',packed)
        outputs, hidden = self.model(packed, hidden)
        #print('before padding')
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        #print('before encoder outputs shape :' , outputs.shape)
        #print('encoder hidden : ',hidden)
        if self.bidirectional:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
            
            
        
        #print('after encoder outputs shape :' , outputs.shape)
        #print('ended')
        return outputs,hidden

class DecoderRNN(nn.Module):
    def __init__(self,input_size,embedding_size,hidden_size,output_size,n_layers = 1,dropout = 0,cell_type = 'lstm',bidirectional = False,activation='relu'):
        super(DecoderRNN,self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.cell_type = cell_type.lower()
        self.activation = activation
        self.embedding = nn.Embedding(input_size,embedding_size)
        if self.cell_type=='rnn':
            self.model = nn.RNN(embedding_size,hidden_size,bidirectional = self.bidirectional,num_layers=self.n_layers, dropout=(0 if n_layers == 1 else dropout) )
            
        elif self.cell_type=='gru':
            self.model = nn.GRU(embedding_size,hidden_size,bidirectional = self.bidirectional,num_layers=self.n_layers, dropout=(0 if n_layers == 1 else dropout) )
        
        elif self.cell_type=='lstm':
            self.model = nn.LSTM(embedding_size,hidden_size,bidirectional = self.bidirectional,num_layers=self.n_layers, dropout=(0 if n_layers == 1 else dropout) )
        
        self.out = nn.Linear(hidden_size,output_size)
        #self.softmax = nn.LogSoftmax(dim=1)
                
        
    def forward(self,input_seq,last_hidden):
        embedded = self.embedding(input_seq)
        activation = get_activation(self.activation)
        output = activation(embedded)
        output, hidden = self.model(output, last_hidden)
        if self.bidirectional:
            output = output[:, :, :self.hidden_size] + output[:, : ,self.hidden_size:]
            
        output = self.out(output)
        #print('before:',output , 'shape = ',output.shape)
        output = F.softmax(output, dim=-1)
        #print('after:',output)
        #output = self.softmax(self.out(output[0]))
        return output, hidden
    


def maskNLLLoss(inp, target, mask):
    #print('1st line')
    #print(inp.shape,target.shape,mask)
    inp = inp.squeeze(0)
    '''if random.random()<0.1:
        print('input = ',inp)
        print('target = ',target)'''
    #print(inp.shape,target.shape,mask)
    nTotal = mask.sum()
    #print('input = ',inp,'target = ', target)
    crossEntropy = -torch.log(torch.gather(inp,1,target.view(-1,1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    #print('it ended')
    return loss,nTotal.item()
        
def mask_accuracy(target,output,mask):
    output = output.int()
    m,n = target.shape
    correct_results = 0
    for i in range(n):
        #print('target shape = ',target[:,i].shape)
        masked_target = target[:,i].masked_select(mask[:,i])
        masked_output = output[:,i].masked_select(mask[:,i])
        '''if random.random()<0.001:
            #print('target = ', target)
            #print('output = ',output)
            print('masked target = ',masked_target)
            print('masked output = ',masked_output)'''
        if torch.equal(masked_target,masked_output):
            #print('------------Correct-Prediction------------------')
            #print('masked target = ',masked_target)
            #print('masked output = ',masked_output)
            #print('------------------------------')
            correct_results+=1
    return correct_results/n
        

def train(input_variable,lengths,target_variable,mask,max_target_len,encoder,decoder,encoder_optimizer,decoder_optimizer,batch_size,clip,max_length = MAX_LENGTH,teacher_forcing_ratio=0.5):
    
    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    # Set device options
    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    m,n = target_variable.shape
    output = torch.zeros((m,n)).to(device)
    
    # Lengths for RNN packing should always be on the CPU
    lengths = lengths.to("cpu")
    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0
    
    # Forward pass through encoder
    #print('train input shape:',input_variable.shape)
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)
    #print('encoder output shape : ',encoder_outputs.shape)
    #print('encoder hidden shape : ',encoder_hidden[0].shape)
    
    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)
    #print('encoder hidden shape',encoder_hidden.shape)
    # Set initial decoder hidden state to the encoder's final hidden state
    
    decoder_hidden = encoder_hidden
    #print('decoder hiidden shape ',decoder_hidden.shape)
    #decoder_hidden = decoder_hidden[:decoder.n_layers]
    #print('decoder hidden shape : ',decoder_hidden.shape)
    
    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    #print('max target length ',max_target_len)
    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            # to calculate accuracy
            _, topi = decoder_output.topk(1)
            #print(topi,batch_size)
            topi = topi.squeeze(2)
            topi = topi.squeeze(0)
            output[t] = torch.LongTensor([topi[i] for i in range(batch_size)])
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            #print('decoder output shape before loss: ',decoder_output.shape)
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            #print(topi,batch_size)
            topi = topi.squeeze(2)
            topi = topi.squeeze(0)
            #print(topi.shape)
            decoder_input = torch.LongTensor([[topi[i] for i in range(batch_size)]])
            output[t] = torch.LongTensor([topi[i] for i in range(batch_size)])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            #print('decoder output shape before loss: ',decoder_output.shape)
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    
    # Perform backpropagation
    #print('before loss.backward')
    loss.backward()

    # Clip gradients: gradients are modified in place
    #print('before gradient clipping')
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)
    
    # Adjust model weights
    #print('before optimizer step')
    encoder_optimizer.step()
    decoder_optimizer.step()
    #print('before accuracy')
    accuracy = mask_accuracy(target_variable,output,mask=mask)
    #print('ended train one batch')
    return (sum(print_losses) / n_totals),accuracy
    
         
         
def trainIters(model_name,pairs,encoder,decoder,encoder_optimizer,decoder_optimizer,encoder_n_layers,decoder_n_layers,save_dir,n_iterations,batch_size,print_every,save_every,clip,teacher_forcing_ratio,input_lang,output_lang,val_pairs):
    
    # Load batches for each iteration
    start = time.time()
    random.seed(42)
    #batch_size = len(val_pairs)
    #n_iterations = 1
    training_batches = [batch2TrainData([random.choice(pairs) for _ in range(batch_size)],input_lang,output_lang) for _ in range(n_iterations)]
    #val_batches = [batch2TrainData([random.choice(val_pairs) for _ in range(batch_size)],input_lang,output_lang) for _ in range(n_iterations)]
    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    print_accuracy = 0
    best_loss = float('inf')
    best_accuracy = 0
    # Training loop
    print("Training...")
    
    for iteration in range(start_iteration,n_iterations+1):
        encoder.train()
        decoder.train()
        training_batch = training_batches[iteration-1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch
        
        # Run a training iteration with batch
        loss ,accuracy = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, batch_size, clip,teacher_forcing_ratio)
        print_loss += loss
        #print('cumulative loss = ',print_loss)
        print_accuracy += accuracy
        # Print progress
        if iteration % print_every == 0:
            encoder.eval()
            decoder.eval()
            print_loss_avg = print_loss / print_every
            print_accuracy_avg = print_accuracy/print_every
            #print('started validation accuracy')
            if iteration%(print_every*5)==0:
                #pass
                val_loss,val_accuracy = evaluateIters(val_pairs,encoder,decoder,input_lang,output_lang)
                print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f} ; Average Train accuracy: {:.4f}; Val loss:{:.4f} ; Val Accuracy:{:.4f} ".format(iteration, iteration / n_iterations * 100, print_loss_avg,print_accuracy_avg*100,val_loss,val_accuracy*100))
                wandb.log({'Iterations':iteration,'Training loss':print_loss_avg,'Training accuracy':print_accuracy_avg*100,'Validation loss':val_loss,'Validation Accuracy':val_accuracy*100})
            else:
                #val_loss,val_accuracy = evaluateIters(val_pairs,encoder,decoder,input_lang,output_lang)
                print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f} ; Average Train accuracy: {:.4f}".format(iteration, iteration / n_iterations * 100, print_loss_avg,print_accuracy_avg*100))
                wandb.log({'Iterations':iteration,'Training loss':print_loss_avg,'Training accuracy':print_accuracy_avg*100})
                
            
            if print_loss_avg<best_loss or print_accuracy_avg>best_accuracy:
                best_loss = print_loss_avg
                torch.save({
                    'iteration': iteration,
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'encoder_optimizer_state_dict': encoder_optimizer.state_dict(),
                    'decoder_optimizer_state_dict': decoder_optimizer.state_dict(),
                    'loss': print_loss_avg,
                    'accuracy': print_accuracy_avg
                },os.path.join('models','%f_accuracy_%s_model.pth'%(print_accuracy_avg*100,model_name)))
            print_loss = 0
            print_accuracy = 0
        end = time.time()
        print('Training time = ',end-start)
        
        # Save checkpoint
        '''if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))'''
    print('Training time = ',end-start)

#def evaluate(encoder,decoder,)
def evaluate(val_input_variable,lengths,val_target_variable,mask,max_target_len,encoder,decoder,batch_size):
    with torch.no_grad():
        #batch_size = len(val_pairs)
        #val_input_variable,lengths,val_output_variable,mask,max_target_len = batch2TrainData(val_pairs,input_lang,output_lang)
        val_input_variable = val_input_variable.to(device)
        val_target_variable = val_target_variable.to(device)
        mask = mask.to(device)
        m,n  = val_target_variable.shape
        output = torch.zeros((m,n)).to(device)
        
        lengths = lengths.to('cpu')
        
        loss = 0
        print_losses = []
        n_totals = 0
        
        # Forward pass through encoder
        #print('before encoder pass in validations')
        #print('input shape:',val_input_variable.shape)
        #print('output shape:',val_output_variable.shape)
        encoder_outputs , encoder_hidden = encoder(val_input_variable,lengths)
        
        # Create initial decoder input (start with SOS tokens for each sentence)
        #print('before decoder input creation ')
        decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
        decoder_input = decoder_input.to(device)
        #print('before decoder hidden creation')
        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden
        #print('max_taget_length ',max_target_len)
        
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            # No teacher forcing
            _, topi = decoder_output.topk(1)
            #print(topi,batch_size)
            topi = topi.squeeze(2)
            topi = topi.squeeze(0)
            
            decoder_input = torch.LongTensor([[topi[i] for i in range(batch_size)]])
            output[t] = torch.LongTensor([topi[i] for i in range(batch_size)])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, val_target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
            
        accuracy = mask_accuracy(val_target_variable,output,mask=mask)

    #return (sum(print_losses) / n_totals),accuracy
    return (sum(print_losses) / n_totals),accuracy

def evaluateIters(val_pairs,encoder,decoder,input_lang,output_lang):
    
    batch_size = 64
    n_iterations = int(len(val_pairs)/batch_size)
    validation_batches = []
    random.shuffle(val_pairs)
    for i in range(n_iterations):
        b = batch2TrainData(val_pairs[i*batch_size:i*batch_size+batch_size],input_lang,output_lang)
        validation_batches.append(b)
    #print('Initializing Validation')
    start_iteration = 1
    print_loss = 0
    print_accuracy = 0
    #print('Validating')
    
    for iteration in range(start_iteration,n_iterations+1):
        encoder.eval()
        decoder.eval()
        val_batch = validation_batches[iteration-1]
        val_input_variable , lengths , val_target_variable,mask,max_target_len = val_batch
        loss,accuracy = evaluate(val_input_variable,lengths,val_target_variable,mask,max_target_len,encoder,decoder,batch_size)
        
        print_loss +=loss 
        print_accuracy += accuracy
    
    print_loss_avg  = print_loss/ n_iterations
    print_accuracy_avg = print_accuracy/n_iterations
    
    return print_loss_avg,print_accuracy_avg      
        
        
        

def train_wandb(config = default_config):
    
    run = wandb.init(project = config.wandb_project , entity = config.wandb_entity,config = config )
    config = wandb.config
    # Configure models
    inp_lang , out_lang = config.input_lang,config.output_lang
    input_lang, output_lang, pairs,val_pairs = prepareData(inp_lang, out_lang) 
    #val_input_lang,val_output_lang,val_pairs = prepareData(inp_lang,out_lang,file='valid')
    hidden_size = config.hidden_size
    encoder_embedding_size = config.encoder_embedding_size
    decoder_embedding_size = config.decoder_embedding_size
    encoder_n_layers = config.n_layers
    decoder_n_layers = config.n_layers
    cell_type = config.cell_type
    batch_size = config.batch_size

    dropout = config.dropout
    activation = config.activation
    bidirectional = True if config.bidirectional=='True' else False
    print(bidirectional)
    print('Building encoder and decoder ...')
    encoder = EncoderRNN(input_lang.n_words, encoder_embedding_size,hidden_size, encoder_n_layers,dropout,cell_type,bidirectional)
    decoder = DecoderRNN(output_lang.n_words,decoder_embedding_size,hidden_size,output_lang.n_words,decoder_n_layers,dropout,cell_type,bidirectional,activation)

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    #checkpoint = torch.load('models/15.851562_accuracy_cell_lstm_e_n_2_d_n_2_h_s_128_b_32_e_lr_0.001000_d_lr_0.001000_n_iter_16000_e_emb_16_d_emb_32_a_relu_model.pth')
    #encoder.load_state_dict(checkpoint['encoder_state_dict'])
    #decoder.load_state_dict(checkpoint['decoder_state_dict'])
    #print('loaded model with 15.81 accuracy')
    # Configure training/optimization
    clip = config.clip
    teacher_forcing_ratio = config.teacher_forcing_ratio
    encoder_learning_rate = config.encoder_learning_rate
    decoder_learning_rate = config.decoder_learning_rate
    n_iteration = config.n_iterations
    print_every = n_iteration/10
    save_every = 500

    #encoder.train()
    #decoder.train()

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = get_optimizer(config.encoder_optimizer,encoder_learning_rate,config.momentum,config.beta,config.beta1,config.beta2,encoder,config.weight_decay)
    decoder_optimizer = get_optimizer(config.decoder_optimizer,decoder_learning_rate,config.momentum,config.beta,config.beta1,config.beta2,decoder,config.weight_decay)

    print("Starting Training!")
    name = 'cell_%s_e_n_%i_d_n_%i_h_s_%i_b_%i_e_lr_%f_d_lr_%f_n_iter_%i_e_emb_%i_d_emb_%i_a_%s'%(cell_type,encoder_n_layers,decoder_n_layers,hidden_size,batch_size,encoder_learning_rate,decoder_learning_rate,n_iteration,encoder_embedding_size,decoder_embedding_size,activation)
    run.name = name
    model_name = name
    trainIters(model_name, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
            encoder_n_layers, decoder_n_layers, 'transliteration_models', n_iteration, batch_size,
            print_every, save_every, clip,teacher_forcing_ratio,input_lang,output_lang,val_pairs)
    
    #evaluate(encoder,decoder,)




if __name__ == '__main__':
    parse_args()
    train_wandb()
    
    