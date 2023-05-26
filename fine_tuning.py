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
from train import Lang,readLangs,prepareData,indexesFromWord,zeroPadding,binaryMatrix,inputVar,outputVar,batch2TrainData,get_activation,get_optimizer,EncoderRNN,DecoderRNN,maskNLLLoss,mask_accuracy,evaluate,evaluateIters

# change it to the below line if running on colab or windows 
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
#for apple silicon with gpu 
device = torch.device('mps')
PAD_token = -1  # Used for padding short sentences
SOS_token = 0  # Start-of-sentence token
EOS_token = 1  # End-of-sentence token
MAX_LENGTH = 24  

default_config = SimpleNamespace(
	wandb_project = 'rnn',
	wandb_entity = 'uttakarsh05',
	n_iterations = 2000, 
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
	#encoder_embedding_size = 8,
	#decoder_embedding_size = 16,
    #n_layers = 2,
    #cell_type = 'lstm',
    #bidirectional = 'False',
    batch_size = 32,
    #hidden_size = 32,
	dropout = 0.2,
    print_every = 50,
	activation = 'relu',
    layer_norm = 'True',
    data_dir = 'data',
    device = 'mps',
    model_path = '',
    freeze = 'encoder'
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
	#argparser.add_argument('-e_emb','--encoder_embedding_size',type = int, default = default_config.encoder_embedding_size , help = 'resolution of input image')
	#argparser.add_argument('-d_emb','--decoder_embedding_size',type = int, default = default_config.decoder_embedding_size , help = 'no of channels in the input image')
	#argparser.add_argument('-n_l','--n_layers',type = int, default = default_config.n_layers , help = 'no of layers in the encoder and decoder')
	#argparser.add_argument('-d_n','--decoder_n_layers',type = int, default = default_config.decoder_n_layers , help = 'no of layers in the decoder')
	#argparser.add_argument('-c','--cell_type',type = str,default = default_config.cell_type,help = 'cell type in encoder and decoder')
	##argparser.add_argument('-bi','--bidirectional',type = str,default = default_config.bidirectional,help = 'if the cell type is bidirectional or not')
	argparser.add_argument('-b','--batch_size',type = int,default = default_config.batch_size,help = 'batch size for training one iterations')
	#argparser.add_argument('-h_s','--hidden_size',type = int,default = default_config.hidden_size,help = 'hidden state size in the encoder and decoder layer')
	argparser.add_argument('-d','--dropout',type = float,default = default_config.dropout,help = 'probability of a neuron to be dropped')
	argparser.add_argument('-p_e','--print_every',type = int,default = default_config.print_every,help = 'no of iterations for calculating training loss')
	argparser.add_argument('-a','--activation',type = str,default = default_config.activation,help = 'activation name')
	argparser.add_argument('-l_n','--layer_norm',type=str , default=default_config.layer_norm, help='layer normalization ')
	argparser.add_argument('-d_dir','--data_dur',type=str , default=default_config.data_dir, help='path to the data folder')
	argparser.add_argument('-dev','--device',type=str , default=default_config.device, help='device used for training the model') 
	argparser.add_argument('-m_p','--model_path',type=str , default=default_config.model_path, help='path to the model for fine tuning') 
	argparser.add_argument('-fr','--freeze',type=str , default=default_config.freeze, help='either encoder or decoder to freeze') 
	args = argparser.parse_args()
	vars(default_config).update(vars(args))
	return 


def train(input_variable,lengths,target_variable,mask,max_target_len,encoder,decoder,encoder_optimizer,decoder_optimizer,batch_size,clip,max_length = MAX_LENGTH,teacher_forcing_ratio=0.5,freeze = 'encoder'):
    
    # Zero gradients
    if freeze =='encoder':
        decoder_optimizer.zero_grad()
        for param in encoder.parameters():
            param.requires_grad = False
    else:
        encoder_optimizer.zero_grad()
        # Freeze the parameters of the decoder
        for param in decoder.parameters():
            param.requires_grad = False
    
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
    #print(teacher_forcing_ratio,type(teacher_forcing_ratio))
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
    if freeze=='encoder':
        decoder_optimizer.step()
    else:
        encoder_optimizer.step()
    #print('before accuracy')
    accuracy = mask_accuracy(target_variable,output,mask=mask)
    #print('ended train one batch')
    return (sum(print_losses) / n_totals),accuracy
    
         
         
def trainIters(model_name,pairs,encoder,decoder,encoder_optimizer,decoder_optimizer,encoder_n_layers,decoder_n_layers,save_dir,n_iterations,batch_size,print_every,save_every,clip,teacher_forcing_ratio,input_lang,output_lang,val_pairs,freeze):
    
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
        if freeze == 'encoder':
            decoder.train()
        else:
            encoder.train()
        #encoder.train()
        #decoder.train()
        training_batch = training_batches[iteration-1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch
        
        # Run a training iteration with batch
        loss ,accuracy = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, batch_size, clip,teacher_forcing_ratio=teacher_forcing_ratio,freeze = freeze)
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
            if iteration%(print_every*2)==0:
                #pass
                val_loss,val_accuracy = evaluateIters(val_pairs,encoder,decoder,input_lang,output_lang)
                print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f} ; Average Train accuracy: {:.4f}; Val loss:{:.4f} ; Val Accuracy:{:.4f} ".format(iteration, iteration / n_iterations * 100, print_loss_avg,print_accuracy_avg*100,val_loss,val_accuracy*100))
                wandb.log({'Iterations':iteration,'Training loss':print_loss_avg,'Training accuracy':print_accuracy_avg*100,'Validation loss':val_loss,'Validation Accuracy':val_accuracy*100})
            else:
                #val_loss,val_accuracy = evaluateIters(val_pairs,encoder,decoder,input_lang,output_lang)
                print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f} ; Average Train accuracy: {:.4f}".format(iteration, iteration / n_iterations * 100, print_loss_avg,print_accuracy_avg*100))
                wandb.log({'Iterations':iteration,'Training loss':print_loss_avg,'Training accuracy':print_accuracy_avg*100})
                
            
            if print_loss_avg<best_loss:
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


def train_wandb(config = default_config):
    
    run = wandb.init(project = config.wandb_project , entity = config.wandb_entity,config = config )
    config = wandb.config
    # Configure models
    inp_lang , out_lang = config.input_lang,config.output_lang
    input_lang, output_lang, pairs,val_pairs = prepareData(inp_lang, out_lang) 
    #val_input_lang,val_output_lang,val_pairs = prepareData(inp_lang,out_lang,file='valid')
    hidden_size = 128
    encoder_embedding_size = 16
    decoder_embedding_size = 32
    encoder_n_layers = 2
    decoder_n_layers = 2
    cell_type = 'lstm'
    batch_size = config.batch_size

    
    dropout = config.dropout
    activation = config.activation
    bidirectional = False
    print(bidirectional)
    print('Building encoder and decoder ...')
    encoder = EncoderRNN(input_lang.n_words, encoder_embedding_size,hidden_size, encoder_n_layers,dropout,cell_type,bidirectional)
    decoder = DecoderRNN(output_lang.n_words,decoder_embedding_size,hidden_size,output_lang.n_words,decoder_n_layers,dropout,cell_type,bidirectional,activation)

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    directory = "models"
    model_files = os.listdir(directory)
    # Filter out any non-model files (e.g., ".DS_Store")
    model_files = [file for file in model_files if not file.startswith(".")]
    sorted_models = sorted(model_files, key=lambda x: float(x[:8]))
    model_pth = sorted_models[-1]
    
    checkpoint = torch.load('models/%s'%(model_pth))
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    print('loaded model with path = ',model_pth)

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
    freeze = config.freeze
    freeze = freeze.lower()
    trainIters(model_name, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
            encoder_n_layers, decoder_n_layers, 'transliteration_models', n_iteration, batch_size,
            print_every, save_every, clip,teacher_forcing_ratio,input_lang,output_lang,val_pairs,freeze)
    
    #evaluate(encoder,decoder,)

if __name__ == '__main__':
    parse_args()
    train_wandb()