export WANDB_PROJECT=""
export WANDB_RUN_GROUP="pretrain"

export WANDB_WATCH="none"
export WANDB_DISABLED="false"
export WANDB_DISABLE_CODE="true"

export TORCH_DISTRIBUTED_DEBUG="DETAIL"
export TORCHDYNAMO_DISABLE="1"

export OMP_NUM_THREADS=2

deepspeed --include=localhost:0,1 --master_port=10234 \
    /root/workspace/llava_next_stage_1.0_train.py \
    --output_dir=/root/output_dir/llava_next/stage1.0 \
    --run_name=llava \
    --cache_dir=/root/.cache/.llava_1.0_preprocess \
    --model_name_or_path=jp1924/Llama2_13b-ClipLarge-Llava \
    --cache_file_name=preprocessor.arrow \
    --preprocessing_batched=true \
    --preprocessing_num_workers=20 \
    --preprocessing_batch_size=1000 \
    --dataset_repo_ls \
        jp1924/KoreanVisionDataforImageDescriptionSentenceExtractionandGeneration \
        jp1924/OutsideKnowledgebasedMultimodalQAData \
        jp1924/VisualQuestionAnswering \
        jp1924/KoLLaVA-CC3M \
    --train_dataset_prefix=train \
    --valid_dataset_prefix=validation \
    --test_dataset_prefix=test \
    --split_valid=true \
    --valid_truncate_num=3000 \
    --per_device_train_batch_size=16 \
    --gradient_accumulation_steps=4 \
    --per_device_eval_batch_size=2 \
    --overwrite_cache=false \
    --num_train_epochs=1 \
    --seed=42 \
    --do_train=true \
    --do_eval=true \
    --do_predict=false \
    --valid_truncate_num=3000 \
    --report_to=wandb \
    --wandb_code_log_dir=/root/workspace/ \
    --learning_rate=1e-3 \
    --lr_scheduler_type=cosine \
    --warmup_ratio=0.03 \
    --weight_decay=0 \
    --evaluation_strategy=steps \
    --save_strategy=steps \
    --save_steps=1000 \
    --eval_steps=1000 \
    --logging_strategy=steps \
    --logging_steps=1 \
    --bf16=true \
    --tf32=true \
    --gradient_checkpointing=true \
    --group_by_length=true \
    --deepspeed=/root/workspace/ds_config/ZeRO_2_act_check.json