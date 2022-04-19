# abbr-exp-ml4h

Implemenation of "Improved Clinical Abbreviation Expansion via Non-Sense-Based Approaches" ([paper](http://proceedings.mlr.press/v136/kim20a.html)), ML4H (Machine Learning for Health) workshop at NeurIPS 2020.

<img src="https://user-images.githubusercontent.com/13655756/101674174-25116f80-3a26-11eb-8161-8f542c573017.png" width="800">

This repository contains the non-sense-based (without gloss) and sense-based (with gloss) approaches to clinical abbreviation expansion based on BERT (The code of the one with permutation language model is coming soon in another repository). The code is based on [BlueBERT](https://github.com/ncbi-nlp/bluebert) (previously named as NCBI-BERT), which is a biomedical version of BERT.

## Prerequisite

1. Tensorflow 1.12+
2. Pre-trained model of BlueBERT
3. A clinical abbreviation expansion dataset (MSH, UMN, or ShARe/CLEF 2013 Task 2)

## How to Run

```
# Install required python packages on your environment
$ pip install -r requirement.txt

# Download the BlueBERT parameters
$ wget https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/NCBI-BERT/NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12.zip
$ unzip NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12.zip -d bert_models

# Download and prepare dataset (UMN / MSH). For the ShARe/CLEF dataset, please run scripts/preprocess_sc13t2.ipynb.
$ ./scripts/download_umn.sh

# Fine-tune and evaulate the model (Masked LM method on UMN, one of 10-fold CV). Similarly for MSH and ShARe/CLEF.
$ ./scripts/umn_masklm2.sh
```

## Acknowledgement

We thank the authors of BERT and BlueBERT for the implementation and the weights pre-trained on biomedical corpora.

## Cite this work

```
@InProceeings{juyong2020improved,
  author    = {Juyong Kim and Linyuan Gong and Justin Khim and Jeremy C. Weiss and Pradeep Ravikumar},
  title     = {Improved Clinical Abbreviation Expansion via Non-Sense-Based Approaches},
  booktitle = {Proceedings of the Machine Learning for Health NeurIPS Workshop (ML4H 2020)}
  year      = {2020}
}
```
