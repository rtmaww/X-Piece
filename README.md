
# X-piece
Our implementation of X-piece. 

**# Searching for Optimal Subword Tokenization in Cross-domain NER (IJCAI 2022)**<br>
*Ruotian Ma, Yiding Tan, Xin Zhou, Xuanting Chen, Di Liang, Sirui Wang, Wei Wu, Tao Gui, Qi Zhang*<br>
https://arxiv.org/abs/2206.03352

## Introduction

**X-piece** is a subword-level algorithm for cross-domain NER tasks.

Reproducing our algorithm may take these 2 steps:

1. **X-piece preparation** : Calculate the optimal tokenization segmentations and save. 
2. **run NER** : run cross-domain NER tasks based on X-piece results. 

## Dependencies
Install dependencies with: 
```bash
pip3 install -r requirements.txt
```

## Dataset
Download distantly/weakly labeled NER data from https://github.com/cliang1453/BOND/tree/master/dataset
Note that all datasets should be preprocessed as **CONLL2003** format. 


## X-piece preparation
For **quick start**, you can just **skip** this process and use the X-Piece result we provide in `ot_datas` to run NER tasks.
X-Piece is a subword-level approach for cross-domain NER, which alleviates the distribution shift between domains directly on the input data via subword-level distribution alignment. 
This **core approach** is implemented in `bpe_ot_new.py`. 
For example: 
```bash
cd utils
mkdir ot_datas/conll2ontonotes/
python3 bpe_ot_new.py \
--label_mode plo \
--source_domain conll --target_domain ontonotes \
--source_path conll --target_path ontonotes \
--subword_data_dir ot_datas/conll2ontonotes/
```  
After this process, the optimal word tokenization distribution of source dataset will be calculated and saved as `ot_datas/conll2ontonotes/ot_list.json` and `ot_datas/conll2ontonotes/ot_ratio.json`, respectively containing the **segmentations** & the **corresponding probabilities**. 

### X-piece Key Arguments
`--label_mode` : label space. "plo" means just including [PER LOC ORG] and "ontonotes" means including all labels in OntoNotes 5.0.  
`--source_domain` : source domain name  
`--target_domain` : target domain name  
`--source_path` : source domain dataset path   
`--target_path` : target domain dataset path   
`--subword_data_dir` : the path to save the xpiece results  
Also, we have the shell script `make_xpiece.sh` as **the demo and for reference**. 

## Run NER
Sample command on how to run NER task with the generated X-piece tokenization distribution:  
```bash
python3 run_ner_ot.py \  
--src conll \  
--tgt ontonotes \  
--task_type NER \  
--train_data_dir conll/ \  
--test_data_dir ontonotes/ \
--log_file log.txt \
--tokenize ot
……
```  

### Run NER Key Arguments
`--src` : same as `--source_domain` in `bpe_ot_new.py`  
`--tgt` : same as `--target_domain` in `bpe_ot_new.py`  
`--train_data_dir` : train dataset directory  
`--test_data_dir` : test dataset directory    
Note that in train dataset directory, `train.txt` is needed, which contains ground-truth labels, while in test dataset directory, `train_lexicon.txt` & `test.txt` is needed, which is the raw text labeled by distant labels & test.  
`--log_file` : the path to save test results  
`--tokenize` : choose whether to use xpiece. "ot" means TO USE and "plain" means NOT.  

Also, we have the shell script `run_ner_conll.sh` as **the demo and for reference**. 

## Summary of Key Folders/Files
- `bpe_eval/`: code for count candidate tokenization methods for every word
- `ner_data/`: dataset dir
- `ot_data/`: save the X-piece results here for further-step use (run NER)
- `tokenizer.py`: the tokenizer we define

## Citation
If you find our repository useful, please consider citing our paper:

```
@article{ma2022searching,
  title={Searching for Optimal Subword Tokenization in Cross-domain NER},
  author={Ma, Ruotian and Tan, Yiding and Zhou, Xin and Chen, Xuanting and Liang, Di and Wang, Sirui and Wu, Wei and Gui, Tao and Zhang, Qi},
  journal={arXiv preprint arXiv:2206.03352},
  year={2022}
}
```


## Acknowledgements
Code is based largely on:
- https://github.com/huggingface/transformers
