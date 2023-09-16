# Transliteration Model
## Report : 
https://api.wandb.ai/links/uttakarsh05/km1zj8ss

## Instructions :

* Clone the repository using  `git clone uttkarsh05/Fodl_assignment_2`

* Create a data folder in the same directory as repository and add all the languages folder there

* To train the seq2seq model from scratch with the default congfiguration run `python train.py`

* To train the attention model from scratch with default configurations run `python train_attention.py`

* After training these two models there will be one model saved in each of the models and attention models directory.

* To fine tune the models saved from the default configurations we will use fine_tuning.py for seq2seq models and fine_tuning_attention.py for attention models.

* Fine tuning the models saved from default configurations from models directory run `python fine_tuning.py`

* Fine tuning the models saved from default configurations from attention_models directory run `python fine_tuning.py`

* If you trained models from train.py or train_attention.py by changing default configurations and you want to fine tune them, just change theese hyperparameters related to encoder decoder architecture

  * cell_type

  * decoder_n_layers

  * encoder_n_layers

  * encoder_embedding_size

  * decoder_embedding_size

  * hidden_size 
  * These params will needed to be changed in the script under train_wandb function in both fine_tuning.py and fine_tuning_attention.py

* To evaluate the best models from models and attention models directory , i have already added one model in both directory which was used to report the test accuracy in the report.

* To test those models run either `python evaluate.py` or `python evaluate_attention.py`.(note runing this will also save those predictions as predictions_vanilla.csv and predictions_attention.csv)

* 4 Sweeps file for training from scratch and fine tuning for both with and without attention have been provided.

* You can also get attention heatmap for the best attention model saved already in attention_models directory.

