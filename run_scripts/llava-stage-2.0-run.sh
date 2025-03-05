export WANDB_PROJECT="MultiModal"
export WANDB_RUN_GROUP='[KoLLaVa9b-patch14-384]stage-2.0'
export WANDB_WATCH=""

export TORCH_DISTRIBUTED_DEBUG="OFF"
export TORCHDYNAMO_DISABLE="1"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,roundup_power2_divisions:[32:256,64:128,256:64,>:32]"
export OMP_NUM_THREADS=2

deepspeed --include=localhost:2,3 --master_port=8532 \
    '/root/workspace/src/main.py' \
    --output_dir="/root/output_dir/KoLLaVa9b-patch14-384/stage2.0" \
    --run_name="llava" \
    --data_preprocessor_type="llava_stage-2.0" \
    --cache_dir="/root/.cache/.[KoLLaVa9b-patch14-384]preprocess/KoLLaVAInsturct" \
    --model_name_or_path="jp1924/KoLLaVa9b-patch14-384-stage1.0" \
    --response_template='[106, 6176, 18145, 235292]' \
    --instruction_template='[106, 6176, 4926, 235292]' \
    --do_data_main_process_first=true \
    --preprocessing_batched=true \
    --preprocessing_num_workers=20 \
    --preprocessing_batch_size=1000 \
    --dataset_repo_ls \
        "jp1924/KoLLaVAInsturct" \
    --train_dataset_prefix="train" \
    --data_max_length=2048 \
    --per_device_train_batch_size=16 \
    --gradient_accumulation_steps=4 \
    --per_device_eval_batch_size=2 \
    --num_train_epochs=2 \
    --seed=42 \
    --do_train=true \
    --do_eval=false \
    --do_predict=false \
    --report_to="none" \
    --learning_rate=1e-3 \
    --lr_scheduler_type="better_cosine" \
    --warmup_ratio=0.03 \
    --weight_decay=0 \
    --eval_strategy="no" \
    --eval_steps=1000 \
    --save_strategy="steps" \
    --save_steps=1000 \
    --logging_strategy="steps" \
    --logging_steps=1 \
    --bf16=true \
    --tf32=true \
    --torch_compile=true \
    --ddp_timeout=1800000000 \
    --dataloader_prefetch_factor=5 \
    --dataloader_num_workers=4 \
    --use_liger_kernel=true \
    --attn_implementation='flash_attention_2' \
    --gradient_checkpointing=true \
    --gradient_checkpointing_kwargs='{"use_reentrant": false}' \
    --config_kwargs='{"vision_feature_select_strateg": "full"}' \
    --processor_kwargs='{"vision_feature_select_strateg": "full"}' \
    --model_kwargs='{"vision_feature_select_strateg": "full"}' \
    --group_by_length=false \
    --packing_max_elem=20 \
    --do_packing=true \
    --packing_shuffle=true \
    --deepspeed="/root/workspace/config/ZeRO_3_act_check.json"