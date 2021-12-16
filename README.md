# HateSpeechDetection

## Dataset overview:
##OffensEval dataset(OLID) readme:

Subtasks:
- a: Offensive, Not offensive
- b: Targeted, Untargeted
- c: Individual | Group | Other

##HatEval dataset readme:

Subtasks:
- a: Hate, Not Hate
- b: Targeted, Untargeted
- c: Aggressive, Not aggresive


## Setup

## Running the scripts:
### Baseline models
#### BERT based basline models 
 1. The notebooks in the following directory contains the code to run BERT and RoBERTa baselines on various subtasks of a,b,c
https://github.com/mrinaltak/HateSpeechDetection/blob/main/Baselines/
 2. The weights for the BERT baseline models for HatEval are present here: 
  HatEval Subtask a: https://drive.google.com/drive/folders/1sUowk-TGXHEprnPHGphkZWTPVnU1x4Lv?usp=sharing
  HatEval Subtask b: https://drive.google.com/drive/folders/1KHpwLVzt7XWi7ieDyDG-ZPtnDaizVuBr?usp=sharing
   HatEval Subtask c: https://drive.google.com/drive/folders/1KtjWE1p__FxXRvDReeXEkrJ1kcSQgoKN?usp=sharing 
   3. The weights for the BERT basline model for OffensEval are present here:
	   OffensEvak subtask a: https://drive.google.com/drive/folders/1vDNGJRMTA7F2eykCgyPdNgXSinFfhYni?usp=sharing

### T-5 small with discrete prompts
1. Notebook to run T-5 model on HatEval is: https://github.com/mrinaltak/HateSpeechDetection/blob/main/T5-discrete/HatEval_t5-discrete.ipynb
2. Notebook to run T-5 model on OffensEval is: https://github.com/mrinaltak/HateSpeechDetection/blob/main/T5-discrete/OffensEval.ipynb
 2. The weights for the T-5 small with discrete prompts for HatEval are present here: 
  HatEval: https://drive.google.com/drive/folders/1NANT7Nwh31lIDx7pge0s2uMKn_udt6FM?usp=sharing
   3. The weights for the T-5 small with discrete prompts for OffensEval are present here: 
  HatEval: https://drive.google.com/drive/folders/16we4QB28e_gM11TsbhF6253VZ-11miLH?usp=sharing
  
### Continuous prompts
- install required packages `pip install -r requirements.txt`
- install transformers from source using -
`pip install git+https://github.com/huggingface/transformers` 
[had to do this because decoding from 'input embeddings' was not available in the last release of transformers but implemented very recently in the official repository, this is needed for continous prompts on T5]
- for training:
  - For OffensEval, use continuous/continuous_prompt.py. Sample launch commands are provided in continuous/launch.sh
  - For Hateval, use continuous/continuous_prompt_hateval.py. Sample launch commands are provided in continuous/launch.sh
- for evaluation, use the respective scripts with `--model_path` argument as the path to the model checkpoint and set `--epochs` to 0. This will run the test script directly.
- for coss evaluation, run the hateval script with the OffensEval checkpoint, and `--n_tasks` as 4.
- model checkpoints can be found [here](https://drive.google.com/drive/folders/1bQlz5FnOSBPgt32SGCnx4qKqy4duoGp5?usp=sharing)
