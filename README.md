# HateSpeechDetection

OLID dataset readme:

All possible label combinations across all the sub-tasks are as below:
- Not offensive
- Offensive, Untargeted
- Offensive, Targeted Insult, (Individual | Group | Other)


## Setup

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