#!/usr/bin/env bash

mkdir data

wget "https://conservancy.umn.edu/bitstream/handle/11299/137703/AnonymizedClinicalAbbreviationsAndAcronymsDataSet.txt" -P data

mkdir -p data/umn_unsupervised
python abbr/create_umn.py data/AnonymizedClinicalAbbreviationsAndAcronymsDataSet.txt data/umn_unsupervised/dev.tsv

mkdir -p data/umn_supervised
python abbr/create_umn.py data/AnonymizedClinicalAbbreviationsAndAcronymsDataSet.txt data/umn_supervised/all.tsv
python abbr/split.py data/umn_supervised/all.tsv data/umn_supervised 0.1 0.1
python abbr/split_10cv.py data/umn_supervised/all.tsv data/umn_supervised
