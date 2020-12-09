#!/usr/bin/env bash

if [  ! -f "data/MSHCorpus.zip" ]; then
    echo "Please download MSH dataset (MSHCorpus.zip) at https://wsd.nlm.nih.gov/collaboration.shtml"
    echo "and place it under data/"
    exit 0
fi

unzip data/MSHCorpus.zip -d data

mkdir -p data/msh_unsupervised_new
python abbr/create_msh_new.py data/MSHCorpus data/msh_unsupervised_new/dev.tsv

mkdir -p data/msh_supervised_new
python abbr/create_msh_new.py data/MSHCorpus data/msh_supervised_new/all.tsv
python abbr/split.py data/msh_supervised_new/all.tsv data/msh_supervised_new 0.1 0.1
python abbr/split_10cv.py data/msh_supervised_new/all.tsv data/msh_supervised_new

# Gloss
mkdir -p data/msh_supervised_gloss
python abbr/create_msh_def.py data/MSHCorpus data/msh_supervised_gloss/all.tsv
python abbr/split.py data/msh_supervised_gloss/all.tsv data/msh_supervised_gloss 0.1 0.1
python abbr/split_10cv.py data/msh_supervised_gloss/all.tsv data/msh_supervised_gloss
