export WANDB_PROJECT="MultiModal"
export WANDB_RUN_GROUP="LLaVa-2.0"


export TORCH_DISTRIBUTED_DEBUG="DETAIL"
export TORCHDYNAMO_DISABLE="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export OMP_NUM_THREADS=2

deepspeed --include=localhost:0,1,2,3 --master_port=10234 \
    '/root/workspace/llava_stage_2.0_train.py' \
    --output_dir='/root/output_dir/KoLLaVa9b-patch14-384/stage2.0' \
    --run_name='llava-instruct' \
    --cache_dir='/root/.cache/.[KoLLaVa9b-patch14-384]preprocessor/stage2.0/KoLLaVAInsturct' \
    --model_name_or_path='jp1924/KoLLaVa9b-patch14-384-stage1.0' \
    --cache_file_name='preprocessor.arrow' \
    --response_template='[106, 6176, 18145, 235292]' \
    --instruction_template='[106, 6176, 4926, 235292]' \
    --do_data_main_process_first=true \
    --preprocessing_batched=true \
    --preprocessing_num_workers=20 \
    --preprocessing_batch_size=1000 \
    --dataset_repo_ls \
        'jp1924/KoLLaVAInsturct' \
    --train_dataset_prefix='train' \
    --per_device_train_batch_size=32 \
    --report_to='wandb' \
    --gradient_accumulation_steps=1 \
    --per_device_eval_batch_size=2 \
    --num_train_epochs=2 \
    --seed=42 \
    --do_train=true \
    --do_eval=false \
    --do_predict=false \
    --learning_rate=1e-5 \
    --lr_scheduler_type='cosine' \
    --warmup_ratio=0.03 \
    --weight_decay=0 \
    --save_strategy=epoch \
    --save_steps=435 \
    --eval_steps=1000 \
    --eval_strategy=no \
    --data_max_length=2048 \
    --logging_strategy=steps \
    --logging_steps=1 \
    --bf16=true \
    --tf32=true \
    --torch_compile=true \
    --use_liger_kernel=true \
    --gradient_checkpointing=true \
    --vision_feature_select_strateg='full' \
    --attn_implementation='flash_attention_2' \
    --gradient_checkpointing_kwargs='{"use_reentrant": false}' \
    --group_by_length=false \
    --dataloader_prefetch_factor=5 \
    --ddp_timeout=1800000000 \
    --dataloader_num_workers=4 \
    --packing_max_elem=20 \
    --do_packing=true \
    --packing_shuffle=true \
    --deepspeed='/root/workspace/ds_config/ZeRO_3_act_check.json'