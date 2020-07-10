# About FasterSeq

As the fast development of the Internet of Things and other mobile devices, the deployment of faster and more efficient deep learning techniques needs more efforts. I create this repository for exploring and developing state-of-the-art accelerating computation and efficient deep learning techniques for faster sequence modeling. The base library we choose is [fairseq](https://github.com/pytorch/fairseq) which is an open-source library for sequence modeling developed and maintained by facebook artificial intelligence research lab.

# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.4.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **To install fairseq** and develop locally:
```bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./
```
* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```
* **For large datasets** install [PyArrow](https://arrow.apache.org/docs/python/install.html#using-pip): `pip install pyarrow`
* If you use Docker make sure to increase the shared memory size either with
`--ipc=host` or `--shm-size` as command line options to `nvidia-docker run`.

<details><summary>List of implemented papers</summary><p>
- **Transformer (self-attention) networks**
  - [Lite Transformer with Long-Short Range Attention (Wu et al., 2020)](https://arxiv.org/abs/2004.11886)
</p></details>

The authors from MIT-Han-Lab have already published their fairseq-based codes, one may check out [here](https://github.com/mit-han-lab/lite-transformer). Additionally, I re-implement the Lite-Transformer since the authors seem used an old version of *fairseq* and it may cause further conflicts especially when you need to apply the *incremenral_state* function.  Before you test the model, please make sure you install the cuda version *lightConv* and *dynamicConv* by:
```bash
cd fairseq/modules/lightconv_layer
python cuda_function_gen.py
python setup.py install
cd fairseq/modules/dynamicconv_layer
python cuda_function_gen.py
python setup.py install
```
Here we use [**IWSLT'14**](http://workshop2014.iwslt.org/downloads/proceeding.pdf) dataset as an example to demo how to train a new Lite-Transformer.

First download and preprocess the data:
```bash
# Download and prepare the data
cd examples/translation/
bash prepare-iwslt14.sh
cd ../..

# Preprocess/binarize the data
TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.de-en \
    --workers 20
```
Next we'll train a new Lite-Transformer translation model over this data:
we choose the *transformer_multibranch_iwslt_de_en* architecture, and save all the training log into 'train.log' file.
```bash
CUDA_VISIBLE_DEVICES=0 fairseq-train \
    data-bin/iwslt14.tokenized.de-en \
    --arch transformer_multibranch_iwslt_de_en \
    --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.2 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 --max-update 50000 \
    --encoder-branch-type attn:1:80:4 lightweight:default:80:4 \
    --decoder-branch-type attn:1:80:4 lightweight:default:80:4 \
    --weight-dropout 0.1 \
    --encoder-embed-dim 160 --decoder-embed-dim 160 \
    --encoder-ffn-embed-dim 160 --decoder-ffn-embed-dim 160 \
    --save-dir checkpoints/transformer_multibranch \
    --tensorboard-logdir checkpoints/transformer_multibranch/log > checkpoints/transformer_multibranch/train.log
```
Finally we can evaluate our trained model:
```bash
fairseq-generate data-bin/iwslt14.tokenized.de-en \
    --path checkpoints/transformer_multibranch/checkpoint_best.pt \
    --batch-size 128 --beam 4 --remove-bpe --lenpen 0.6
```

Later on, I will focus on all the state-of-the-art faster sequence modeling techniques, including re-implementation, validating and try to give a track on all the effective techniques to make efficient computation and accelerating in deep learning based on *fairseq* (**More is Comming..** )



# Getting Started

The [full documentation](https://fairseq.readthedocs.io/) contains instructions for getting started, training new models and extending fairseq with new model types and tasks. Fore complete details, please go check the fairseq website https://github.com/pytorch/fairseq.

# Pre-trained models and examples

**fairseq** provides pre-trained models and pre-processed, binarized test sets for several tasks listed below, as well as example training and evaluation commands.

- [Translation](examples/translation/README.md): convolutional and transformer models are available
- [Language Modeling](examples/language_model/README.md): convolutional and transformer models are available

We also have more detailed READMEs to reproduce results from specific papers:
- [Training with Quantization Noise for Extreme Model Compression](examples/quant_noise/README.md)
- [Neural Machine Translation with Byte-Level Subwords (Wang et al., 2020)](examples/byte_level_bpe/README.md)
- [Multilingual Denoising Pre-training for Neural Machine Translation (Liu et at., 2020)](examples/mbart/README.md)
- [Jointly Learning to Align and Translate with Transformer Models (Garg et al., 2019)](examples/joint_alignment_translation/README.md )
- [Levenshtein Transformer (Gu et al., 2019)](examples/nonautoregressive_translation/README.md)
- [Facebook FAIR's WMT19 News Translation Task Submission (Ng et al., 2019)](examples/wmt19/README.md)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al., 2019)](examples/roberta/README.md)
- [wav2vec: Unsupervised Pre-training for Speech Recognition (Schneider et al., 2019)](examples/wav2vec/README.md)
- [Mixture Models for Diverse Machine Translation: Tricks of the Trade (Shen et al., 2019)](examples/translation_moe/README.md)
- [Pay Less Attention with Lightweight and Dynamic Convolutions (Wu et al., 2019)](examples/pay_less_attention_paper/README.md)
- [Understanding Back-Translation at Scale (Edunov et al., 2018)](examples/backtranslation/README.md)
- [Classical Structured Prediction Losses for Sequence to Sequence Learning (Edunov et al., 2018)](https://github.com/pytorch/fairseq/tree/classic_seqlevel)
- [Hierarchical Neural Story Generation (Fan et al., 2018)](examples/stories/README.md)
- [Scaling Neural Machine Translation (Ott et al., 2018)](examples/scaling_nmt/README.md)
- [Convolutional Sequence to Sequence Learning (Gehring et al., 2017)](examples/conv_seq2seq/README.md)
- [Language Modeling with Gated Convolutional Networks (Dauphin et al., 2017)](examples/language_model/conv_lm/README.md)


