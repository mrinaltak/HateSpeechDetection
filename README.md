# HateSpeechDetection

OLID dataset readme:

All possible label combinations across all the sub-tasks are as below:
  Not offensive
  Offensive, Untargeted
  Offensive, Targeted Insult, (Individual | Group | Other)


## Setup

- install required packages `pip install -r requirements.txt`
- install transformers from source 
`pip install git+https://github.com/huggingface/transformers` 
[had to do this because decoding from 'input embeddings' was not available in the last release of transformers but implemented very recently in the official repository, this is needed for continous prompts on T5]
