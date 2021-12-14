python continuous_prompt.py --epochs 4 --lr 1e-5  
python continuous_prompt.py --epochs 3 --lr 0.01 --model T5 --n_tokens 20 
python continuous_prompt.py --epochs 3 --lr 0.03 --model ELECTRA --lr_decay 0.5 --prefix FixCont >outputs/f_electra_performance1.txt
python continuous_prompt.py --epochs 4 --lr 0.03 --model BERT --lr_decay 0.5 --prefix FixCont >outputs/f_bert_performance1.txt

##test these models
python continuous_prompt.py --epochs 0 --lr 0.01 --model BERT --lr_decay 0.5 --model_path Continuous_BERT_20_0.01_8_0.5/ckpt_epoch_3.pth >bert_performance.txt
python continuous_prompt.py --epochs 0 --lr 0.01 --model ELECTRA --lr_decay 0.5 --model_path FixCont_ELECTRA_20_0.01_8_0.5/ckpt_epoch_2.pth >outputs/f_electra_performance1.txt
python continuous_prompt.py --epochs 0 --lr 0.01 --model T5 --lr_decay 0.5 --n_tokens 40 --model_path Continuous_T5_40_0.003_8/ckpt_epoch_2.pth >t5_performance.txt
python continuous_prompt.py --epochs 0 --lr 0.01 --model T5 --lr_decay 0.5 --n_tokens 20 --model_path Continuous_T5_20_0.01_8_1/ckpt_epoch_2.pth >t5_performance_20.txt

##Hateval
python continuous_prompt_hateval.py --epochs 3 --lr 0.01 --model T5 >outputs/hateval_t5.txt
python continuous_prompt_hateval.py --epochs 3 --lr 0.01 --model ELECTRA --lr_decay 0.5 --prefix FixCont >outputs/f_hateval_electra.txt
python continuous_prompt_hateval.py --epochs 3 --lr 0.01 --model BERT --lr_decay 0.5 --prefix FixCont >outputs/f_hateval_bert.txt

##test these models
python continuous_prompt_hateval.py --epochs 0 --lr 0.01 --model BERT --lr_decay 0.5 --model_path HatEval_Continuous_BERT_20_0.01_8_0.5/ckpt_epoch_2.pth >outputs/hateval_bert_performance.txt
python continuous_prompt_hateval.py --epochs 0 --lr 0.01 --model ELECTRA --lr_decay 0.5 --model_path HatEval_Continuous_ELECTRA_20_0.01_8_0.5/ckpt_epoch_2.pth >outputs/hateval_electra_performance.txt
python continuous_prompt_hateval.py --epochs 0 --lr 0.01 --model T5 --model_path HatEval_Continuous_T5_20_0.01_8_1/ckpt_epoch_2.pth >outputs/hateval_t5_performance.txt


##cross eval
python continuous_prompt_hateval.py --epochs 0 --lr 0.01 --model ELECTRA --lr_decay 0.5 --n_tasks 4 --prefix cross --model_path Continuous_ELECTRA_20_0.01_8_0.5/ckpt_epoch_3.pth >outputs/cross_electra.txt
python continuous_prompt_hateval.py --epochs 0 --lr 0.01 --model BERT --lr_decay 0.5 --n_tasks 4 --prefix cross --model_path Continuous_BERT_20_0.01_8_0.5/ckpt_epoch_3.pth >outputs/cross_bert.txt
python continuous_prompt_hateval.py --epochs 0 --lr 0.01 --model T5 --lr_decay 0.5 --n_tasks 4 --n_tokens 40 --prefix cross --model_path Continuous_T5_40_0.003_8/ckpt_epoch_2.pth >outputs/cross_t5.txt
python continuous_prompt_hateval.py --epochs 0 --lr 0.01 --model T5 --lr_decay 0.5 --n_tasks 4 --n_tokens 20 --prefix cross --model_path Continuous_T5_20_0.01_8_1/ckpt_epoch_2.pth >outputs/cross_t5_20.txt


python continuous_prompt_hateval.py --epochs 0 --lr 0.01 --model ELECTRA --lr_decay 0.5 --n_tasks 4 --prefix cross --model_path Continuous_ELECTRA_20_0.01_8_0.5/ckpt_epoch_3.pth >test.txt
