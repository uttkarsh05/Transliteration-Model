from train import default_config as config
from train import Lang,EncoderRNN,DecoderRNN,mask_accuracy,maskNLLLoss,batch2TrainData
import torch,random
import pandas as pd

device = torch.device('mps')

PAD_token = -1  # Used for padding short sentences
SOS_token = 0  # Start-of-sentence token
EOS_token = 1  # End-of-sentence token
MAX_LENGTH = 24 

def readLangs(lang1,lang2):
    lines = open('data/%s/%s_train.csv' % (lang2, lang2), encoding='utf-8').read().strip().split('\n')
    pairs = [[word for word in l.split(',')] for l in lines]
    val_lines = open('data/%s/%s_test.csv' % (lang2, lang2), encoding='utf-8').read().strip().split('\n')
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

input_lang, output_lang, pairs,test_pairs = prepareData('eng', 'hin')
#output_lang.addLetter('UKN')

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
encoder.to(device)
decoder.to(device)

model_pth = 'models/23.265625_accuracy_cell_lstm_e_n_2_d_n_2_h_s_128_b_64_e_lr_0.000100_d_lr_0.000100_n_iter_2000_e_emb_16_d_emb_32_a_relu_model.pth'
checkpoint = torch.load('%s'%(model_pth))
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])
print('loaded model with path = ',model_pth)
predictions = pd.DataFrame(columns=['input','target','predicted'])

def prediction(input_variable,target_variable,output,mask,input_lang,output_lang):
    m,n = input_variable.shape
    m_o,n = output.shape
    
    for j in range(n):
        input_str = ''
        target_str = ''
        predicted_str = ''
        for i in input_variable[:,j]:
            if i.item()==EOS_token or i.item==PAD_token:
                break
            input_str+= input_lang.index2letter[i.item()]
        
        for k in target_variable[:,j]:
            if k.item()==EOS_token or k.item()==PAD_token:
                break
            target_str+= output_lang.index2letter[k.item()]
        
        for l in output[:,j]:
            if l.item()==EOS_token:
                break
            predicted_str+= output_lang.index2letter[l.item()]
        
        predictions.loc[len(predictions)] = [input_str,target_str,predicted_str]
    
        
        
        
            
        
        
            
    


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
        prediction(val_input_variable,val_target_variable,output,mask,input_lang,output_lang)
        
    #return (sum(print_losses) / n_totals),accuracy
    return (sum(print_losses) / n_totals),accuracy

def evaluateIters(val_pairs,encoder,decoder,input_lang,output_lang):
    
    batch_size = 64
    n_iterations = int(len(val_pairs)/batch_size)
    validation_batches = []
    #random.shuffle(val_pairs)
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
        

#test_loss,test_accuracy = evaluateIters(test_pairs,encoder,decoder,input_lang,output_lang)
#print('Test loss : ',test_loss, 'Test Accuracy : ',test_accuracy*100)

if __name__ == '__main__':
    
    test_loss,test_accuracy = evaluateIters(test_pairs,encoder,decoder,input_lang,output_lang)
    print('Test loss : ',test_loss, 'Test Accuracy : ',test_accuracy*100)
    predictions.to_csv('predictions_vanilla.csv')
    #print(input_lang.n_words) 
