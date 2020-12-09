#!/bin/bash
export PYTHONPATH=.

exp_base=exp_msh_masklm2_new
cv_idx=0  # 0~9 for 10-fold cross validation

beta_mask_arr=(0.1 0.1 0.5 0.5 2.0 2.0 5.0 5.0)
lr_arr=(2e-5 5e-5 2e-5 5e-5 2e-5 5e-5 2e-5 5e-5)
best_hps=(6 8 6 8 4 6 6 6 4 4)
hp=${best_hps[$cv_idx]}

exp=${exp_base}_cv${cv_idx}
#beta_mask=2.0
#lr=5e-5
beta_mask=${beta_mask_arr[$hp-1]}
lr=${lr_arr[$hp-1]}
data_dir=data/msh_supervised_new
train_dataset=traindev
eval_dataset=test

python abbr/run_abbr_masklm2.py \
    --data_dir=${data_dir} \
    --bert_config_file=bert_models/bert_config.json \
    --vocab_file=bert_models/vocab.txt \
    --output_dir=results/$exp \
    --init_checkpoint=bert_models/bert_model.ckpt \
    --do_lower_case=True \
    --batch_size=16 \
    --max_n_masks=16 \
    --do_train=True \
    --cv_idx=${cv_idx} \
    --dataset=${train_dataset} \
    --learning_rate=$lr \
    --num_train_epochs=5 \
    --loss_margin=$beta_mask \
    --loss_cross_entropy=0.0 \
    --save_checkpoints_steps=9999999

python abbr/run_abbr_masklm2.py \
    --data_dir=${data_dir} \
    --bert_config_file=bert_models/bert_config.json \
    --vocab_file=bert_models/vocab.txt \
    --output_dir=results/${exp}_out \
    --init_checkpoint=results/${exp} \
    --do_lower_case=True \
    --do_train=False \
    --cv_idx=${cv_idx} \
    --dataset=${eval_dataset} \
    --batch_size=16 \
    --max_n_masks=16

echo "beta_mask=$beta_mask lr=$lr exp=$exp"
python abbr/evaluate.py results/${exp}_out \
    --loss_similarity=0.0
