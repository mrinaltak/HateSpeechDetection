python continuous_prompt.py --epochs 3 --lr 0.01 --model T5 --n_tokens 20 
python continuous_prompt.py --epochs 3 --lr 1e-5 --model ELECTRA --lr_decay 0.5 --prefix FixCont >outputs/electra_performance.txt
python continuous_prompt.py --epochs 4 --lr 0.03 --model BERT --lr_decay 0.5 --prefix FixCont >outputs/bert_performance.txt

##Hateval
python continuous_prompt_hateval.py --epochs 3 --lr 0.01 --model T5 >outputs/hateval_t5.txt
python continuous_prompt_hateval.py --epochs 3 --lr 1e-5 --model ELECTRA --lr_decay 0.5 >outputs/hateval_electra.txt
python continuous_prompt_hateval.py --epochs 3 --lr 0.01 --model BERT --lr_decay 0.5 >outputs/hateval_bert.txt

##cross eval
python continuous_prompt_hateval.py --epochs 0 --model ELECTRA --n_tasks 4 --prefix cross --model_path Continuous_ELECTRA/ckpt_epoch_3.pth >outputs/cross_electra1.txt
python continuous_prompt_hateval.py --epochs 0 --model BERT --n_tasks 4 --prefix cross --model_path Continuous_BERT/ckpt_epoch_3.pth >outputs/cross_bert.txt
python continuous_prompt_hateval.py --epochs 0 --model T5 --n_tasks 4 --n_tokens 20 --prefix cross --model_path Continuous_T5_20_0.01_8_1/ckpt.pth >outputs/cross_t5.txt
