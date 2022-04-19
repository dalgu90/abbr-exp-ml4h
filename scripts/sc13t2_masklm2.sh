#!/bin/bash
export PYTHONPATH=.

exp_base=exp_sc13t2_masklm2

beta_mask_arr=(0.1 0.1 0.5 0.5 2.0 2.0 5.0 5.0)
lr_arr=(2e-5 5e-5 2e-5 5e-5 2e-5 5e-5 2e-5 5e-5)
hp=6

exp=${exp_base}
beta_mask=${beta_mask_arr[$hp-1]}
lr=${lr_arr[$hp-1]}
data_dir=data/sc13t2

python abbr/run_abbr_masklm2_cui.py \
    --data_dir=${data_dir} \
    --bert_config_file=bert_models/bert_config.json \
    --vocab_file=bert_models/vocab.txt \
    --output_dir=results/$exp \
    --init_checkpoint=bert_models/bert_model.ckpt \
    --do_lower_case=True \
    --batch_size=16 \
    --max_n_masks=16 \
    --do_train=True \
    --dataset=train \
    --learning_rate=$lr \
    --num_train_epochs=5 \
    --loss_margin=$beta_mask \
    --loss_cross_entropy=0.0 \
    --save_checkpoints_steps=9999999

python abbr/run_abbr_masklm2_cui.py \
    --data_dir=${data_dir} \
    --bert_config_file=bert_models/bert_config.json \
    --vocab_file=bert_models/vocab.txt \
    --output_dir=results/${exp}_out \
    --init_checkpoint=results/${exp} \
    --do_lower_case=True \
    --do_train=False \
    --dataset=test \
    --batch_size=16 \
    --max_n_masks=16

echo "beta_mask=$beta_mask lr=$lr exp=$exp"
python abbr/evaluate_cui.py results/${exp}_out \
    --test_file_path ${data_dir}/test.tsv
