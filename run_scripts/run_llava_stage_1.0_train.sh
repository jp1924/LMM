export WANDB_PROJECT="MultiModal"
export WANDB_RUN_GROUP="LLaVa"

export TORCH_DISTRIBUTED_DEBUG="DETAIL"
export TORCHDYNAMO_DISABLE="1"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"

export OMP_NUM_THREADS=2


deepspeed --num_gpus=1 \
    '/root/workspace/src/main.py' \
    --output_dir='/root/output_dir/KoLLaVa9b-patch14-384/1.stage1.0' \
    --run_name='llava preprocessing' \
    --cache_dir='/root/.cache/.[KoLLaVa9b-patch14-384]preprocessor/stage1.0' \
    --cache_file_name='preprocessor.arrow' \
    --model_name_or_path='/root/output_dir/KoLLaVa9b-patch14-384' \
    --preprocessing_batched=true \
    --preprocessing_num_workers=20 \
    --preprocessing_batch_size=1000 \
    --dataset_repo_ls \
        'jp1924/KoLLaVA-CC3M' \
        'jp1924/Laion2BMultiKoreanSubset' \
        'jp1924/Coyo700m-1' \
    --data_truncate_map='{"jp1924/Laion2BMultiKoreanSubset": {"train": 158000}, "jp1924/KoLLaVA-CC3M": {"train": 300000}, "jp1924/Coyo700m-1": {"train": 100000}}' \
    --train_dataset_prefix='train' \
    --per_device_train_batch_size=6 \
    --gradient_accumulation_steps=1 \
    --per_device_eval_batch_size=2 \
    --do_data_main_process_first=true \
    --num_train_epochs=1 \
    --seed=42 \
    --do_train=true \
    --do_eval=false \
    --do_predict=false \
    --report_to='wandb' \
    --learning_rate=1e-3 \
    --lr_scheduler_type='cosine' \
    --warmup_ratio=0.03 \
    --weight_decay=0 \
    --eval_strategy='no' \
    --eval_steps=1000 \
    --save_strategy='steps' \
    --save_steps=2000 \
    --logging_strategy='steps' \
    --logging_steps=1 \
    --bf16=true \
    --tf32=true \
    --ddp_timeout=1800000000 \
    --use_liger_kernel=true \
    --gradient_checkpointing=true \
    --gradient_checkpointing_kwargs='{"use_reentrant": false}' \
    --sot_token="<start_of_turn>" \
    --eot_token="<end_of_turn>" \
    --response_template='[6176, 18145, 235292, 108]' \
    --vision_feature_select_strategy='full' \
    --torch_compile=true \
    --use_liger_kernel=true \
    --group_by_length=false \
    --data_max_length=1000 \
    --remove_unused_columns=false \
    --torch_compile=true \
    --torch_empty_cache_steps=100 \
    --deepspeed='/root/workspace/config/ZeRO_3_act_check.json'