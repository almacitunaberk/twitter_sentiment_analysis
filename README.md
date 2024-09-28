# Batch Ensemble of Tokens for Twitter Sentiment Analysis
**Tunaberk Almaci,** **Steve Rhyner,** **Yustina Ovcharova,**

## Setup

After cloning the repository, please create a virtual environment and run the following script:
```
pip install -r requirements.txt
```

## Preprocessing
The non-deep baseline models Logistic Regression, Ridge Rigression, Random Forest Classifier, and Bernoulli Naive Bayes Classifier are trained with preprocessed data.
To create the preprocessed version of the training data and test data, please create a data directory with the training and test data inside it.
If you have a different directory structure, please use the absolute path of the positive/negative training data as well as the test data inplace of the pos_input_path ,neg_input_path, and test_input_ath arguments.
```
cd ./preprocess
python preprocessing.py --pos_input_path=../data/train_pos_full.txt --neg_input_path=../data/train_neg_full.txt --test_input_path=../data/test_data.txt --output_dir=../data/csv/ --output_filename=processed_train --output_test_filename=processed_test --lowercase=True --no_url=True --no_user=True --no_numbers=True --no_extra_space=True --no_stopwords=True --soft_lem=True --slang_conv=True
```

If you want to train with a specific set of preprocessing rules, you can use the predefined arguments together with the above arguments:

*--lowercase*: To use only lowercase letters. Default is False  
*--no_url*: To remove the **<url>** tag from the tweets. Default is False  
*--no_user*: To remove the **<user>** tag from the tweets. Default is False  
*--no_hashtag*: To remove the hashtags from the tweets. Default is False  
*--no_numbers*: To remove the numbers from the tweets. Default is False  
*--no_extra_space*: To remove the extra space from the tweets. Default is False  
*--no_stopwords*: To remove the stop words from the tweets. Default is False  
*--soft_lem*: To apply soft lemmatizatation to the words. Default is False  
*--hard_lem*: To apply hard lemmatization to the words. Default is False. Only apply this if you believe hard lemmatization would improve the model accuracy.  
*--stemming*: To apply stemming to the words. Default is False.  
*--slang_conv*: To apply conversion of common slangs in the tweets. Default is False.  
*--parallelize*: To parallelize the work across cores. Default is False. Parallelizing is highly recommended for multicore architectures.  
*--num_parallels*: If *parallelize* is used, number of parallel threads to perform parallelization.  

The deep models (i.e. BiLSTM model and pretraining of the large language models) are trained with raw data. To produce the CSV file for the unprocessed/raw training and test data, we recommend running the following script:
```
cd ./preprocess
python preprocessing.py --pos_input_path=../data/train_pos_full.txt --neg_input_path=../data/train_neg_full.txt --test_input_path=../data/test_data.txt --output_dir=../data/processed/ --output_filename=raw_train --output_test_filename=raw_test
```

## Training Baseline Models
### Non-Deep Baseline Models

To train the non-deep baseline models, the following script trains all non-deep models in one go:
```
cd ./baseline_non_deep_models
python preprocessing.py --input_path=../data/csv/processed_train.csv --log_path=logs --log_filename=non_deep_logs --save_path=saved_models
```
*--log_path* and *--log_filename* are used as the path that will contain the logs for the models. Logs include the training accuracy as well as the validation accuracy.

### BiLSTM Model

To train the BiLSTM baseline model, please create a config file under /baseline_deep_models/configs. To reproduce the same results, please use the provided config.yaml file at ./baseline_deep_models/configs/config.yaml. But, input_path, log_path, log_filename, and save_path have to be changed to desired paths. **Please set use_wandb under general to False if you do not have a Weights&Biases account**.

```
cd ./baseline_deep_models
python train.py --config_path=./baseline_deep_models/configs/config.yaml
```

If you want to train with different arguments, here are the description of the parameters:

**general**:  
&nbsp;&nbsp; *input_path* Path of the input file. Should be the CSV file obtained from preprocessing  
&nbsp;&nbsp; *batch_size* The batch size to be used during training  
&nbsp;&nbsp; *seed* The seed for reproducability. Default used one is 42 
&nbsp;&nbsp; *use_wandb* The experiments are run with Weights&Biases account. If you don't have a W&B account, please set this to False. **If you are using WandB, you are supposed to enter *project* and *entity* values under general as well**  
&nbsp;&nbsp; *log_path* The path to the log file for logging the training and validation accuracies. If you are using Weights&Biases for monitoring, you can set this to a dummy value  
&nbsp;&nbsp; *save_path* Path where the trained model should be saved.  
&nbsp;&nbsp; *dataloader_workers* Number of workers to be used for the DataLoader of torch.  
&nbsp;&nbsp; *max_epochs* The maximum number of epochs to train the model. We observed that after 1 epoch, the validation and training accuracy become stable. Thus, the default value for max_epochs is 3  
&nbsp;&nbsp; *run_id* The run_id to be used with Weights&Biases.  
&nbsp;&nbsp; *debug* Set this to True to run only to check errors. Debug mode uses only 10,000 samples from the training data.  
&nbsp;&nbsp; *validation_size* The validation percentage of the train/validation data split.  

**model**:  
&nbsp;&nbsp; *glove_size* The size of the pretrained GloVe vectors to be used. Default is 100.  
&nbsp;&nbsp; *drop_prob* The dropout probability to be used during training to increase the generalization capacity of the model.  
&nbsp;&nbsp; *lr* Initial learning rate for the model. Default is 0.01  

## Finetuning Large Language Models

To finetune each large language model used, we provided sample config.yaml files under the configs directory. If you wish to change configuration of them, you can do so manually in each config file.

**Important:** To reproduce the same results, the data_path under general in each config file should be changed to the **absolute path of the CSV file containing the training data**. 

To train a model, please run the following script:
```
cd ./finetuning
python train.py --config_path=./configs/cardiff-base.yaml
```

The trained model will be saved to a directory called saved_models under the src folder. 
You can change the name of the saved file by the **save_name** parameter in the config file under the model.

## Training the Ensemble Model

To train an ensemble model using the CLS tokens generated by: *cardiff-base* and *vinai-base*, please run the following scripts in order

```
cd ./novelty
python save_cls_tokens.py --config_path=./configs/cardiff-base-train-cls-tokens.yaml
python save_cls_tokens.py --config_path=./configs/vinai-base-train-cls-tokens.yaml
```

This script will extract the CLS token generated by each specified model in a .npy file. These tokens are used by the ensemble model as training data.

To train the ensemble model itself, please run the following script 
```
cd ./novelty
python ensemble_train.py --config_path=./configs/ensemble-train-config.yaml
```

The trained models will be saved under the saved_models folder. The script trains different ensemble models with different threshold parameters.

To test the ensemble model, first extract the CLS tokens corresponding to the test data:
```
cd ./novelty
python save_cls_tokens.py --config_path=./configs/cardiff-base-test-cls-tokens.yaml
python save_cls_tokens.py --config_path=./configs/vinai-base-test-cls-tokens.yaml
```

Then, run the following script:
```
cd ./novelty
python ensemble_test.py --config_path=./configs/ensemble-test-config.yaml
```

## Details

The experiments are executed on NVIDIA Tesla V100 GPUs on AWS. Running the provided scripts on different hardware should not produce different results, but they are expected to take longer on older hardware.

