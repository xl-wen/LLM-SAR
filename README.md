# LLM-SAR

This is the PyTorch implementation for LLM-SAR: LLM-Driven Soft-Alignment and Self-Alignment Learning for Recommendation.
>  A portion of our implementation has been open-sourced. We will release the full code once the paper is accepted.

# Environment
- Anaconda 3
- python 3.8
- pytorch 1.13.1

# How to run the codes
```
python encoder/train_encoder.py --model {model_name} --dataset {dataset} --cuda 0

-LLM-SAR

python encoder/train_encoder.py --model lightgcn_cil --dataset amazon --cuda 0
=
```

The structure of this code is based on [IRLLRec](https://github.com/wangyu0627/IRLLRec) and [RLMRec](https://github.com/HKUDS/RLMRec). Thanks for their excellent work!
