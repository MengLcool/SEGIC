NGPUS=${1:-8}
encoder_model=${2:-'dinov2'}
exp_name=${3:-'OUTPUT/all_exps/abs_backbone/dinov2_l'}

echo $NGPUS $encoder_model $exp_name
echo ${@:4}

python -m torch.distributed.launch --master_port 12345 --nproc_per_node=$NGPUS train.py --output $exp_name \
    --input_keys sem_corr point --eval_keys sem_corr --noised_inst  --use_dual_aug --use_simm_prompt --open_ft --find_unused_params --use_dift  \
    --use_inst_proj --diff_text_prompt_ratio 0.75  --use_inst_train --reverse_context --learning_rate 0.0001 --use_cross_inst_prompt \
    --encoder_model $encoder_model --inst_datasets coco lvis --sem_datasets coco ade20k --samples_per_epoch 80000 --auto_resume ${@:4}
