# GAPA

Pytorch Implementation for our paper "Automatic Context Pattern Generation for Entity Set Expansion" in TKDE 2023.



## Prerequisites

python >= 3.7.9

pytorch >= 1.8.1



## Data

**This work does not require a special ESE training corpus, and the training corpus is the same as the original training method of GPT-2 which only requires a huge amount of unsupervised text.** 

The raw corpus we use for pre-training the reverse GPT-2 model can be directly downloaded from https://archive.org/details/enwiki-20171201 .



## Data Preprocessing

We only keep the first ten words in the prev-text of entities and then reverse it. The result will be saved as "sentences-reverse.txt" for the reverse GPT-2 training.

The format looks like this:

| the raw sentence               | the result                     |
| ------------------------------ | ------------------------------ |
| W1 W2 W3 W4 W5 W6 W7 W8 W9 W10 | W10 W9 W8 W7 W6 W5 W4 W3 W2 W1 |



## GPT2_Train

To pre-train a reverse GPT-2 model, run
```
nohup python -m torch.distributed.launch --nproc_per_node=4 run_clm.py --do_train --block_size 128 --per_device_train_batch_size 50 --per_device_eval_batch_size 20 --output_dir './' --model_type gpt2 --tokenizer_name 'gpt2' --train_file  sentences-reverse.txt --num_train_epochs 200000 --save_steps 1000 --fp16 true --ddp_find_unused_parameters false --sharded_ddp "zero_dp_3 auto_wrap" &
```



## GPT2_Generate

Run

```
python generate.py -dataset ./wiki
```

to use the reverse GPT-2 model for generating prev-text from right to left and then to use the regular GPT-2 for generating  next-text.

The result will be saved as "sentence.json".



## GAPA

To get the context representations, run
```
python PretrainedEmb.py -dataset ./wiki
```

To finish the entity set expansion, run
```
python expan.py -dataset ./wiki -output ./wiki/results
```

Expansion result will be saved under the path set.



## Citation

If you consider our paper or code useful, please cite our paper:

```
@article{li2022automatic,
  title={Automatic Context Pattern Generation for Entity Set Expansion},
  author={Li, Yinghui and Huang, Shulin and Zhang, Xinwei and Zhou, Qingyu and Li, Yangning and Liu, Ruiyang and Cao, Yunbo and Zheng, Hai-Tao and Shen, Ying},
  journal={arXiv preprint arXiv:2207.08087},
  year={2022}
}
```

