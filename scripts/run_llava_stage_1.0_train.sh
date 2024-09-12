export COMET_PROJECT_NAME="llava"

export TORCH_DISTRIBUTED_DEBUG="DETAIL"
export TORCHDYNAMO_DISABLE="1"

export OMP_NUM_THREADS=2

deepspeed --include=localhost:0,1,2,3 --master_port=10234 \
    /root/workspace/llava_stage_1.0_train.py \
    --output_dir=/root/output_dir/llava/stage-1 \
    --run_name=llava-stage-1.0 \
    --cache_dir=/root/.cache/.llava_stage_1.0_preprocess \
    --model_name_or_path=/root/output_dir/llava/KoLLaVA-7b \
    --cache_file_name=preprocessor.arrow \
    --preprocessing_batched=true \
    --preprocessing_num_workers=20 \
    --preprocessing_batch_size=1000 \
        jp1924/KoreanVisionDataforImageDescriptionSentenceExtractionandGeneration \
        jp1924/OutsideKnowledgebasedMultimodalQAData \
        jp1924/VisualQuestionAnswering \
        jp1924/KoLLaVA-CC3M \
        jp1924/Laion2BMultiKoreanSubset \
        jp1924/Coyo700m-1 \
    --data_truncate_map='{"jp1924/KoreanVisionDataforImageDescriptionSentenceExtractionandGeneration": {"train": 2000000}, "jp1924/OutsideKnowledgebasedMultimodalQAData": {"train": 500000}, "jp1924/VisualQuestionAnswering": {"train": 500000}, "jp1924/KoLLaVA-CC3M": {"train": 500000}, "jp1924/Laion2BMultiKoreanSubset": {"train": 500000}, "jp1924/Coyo700m-1": {"train": 2000000}}' \
    --train_dataset_prefix=train \
    --per_device_train_batch_size=16 \
    --gradient_accumulation_steps=4 \
    --per_device_eval_batch_size=2 \
    --num_train_epochs=1 \
    --seed=42 \
    --do_train=true \
    --do_eval=false \
    --do_predict=false \
    --report_to=comet_ml \
    --learning_rate=1e-3 \
    --lr_scheduler_type=cosine \
    --warmup_ratio=0.03 \
    --weight_decay=0 \
    --save_strategy=steps \
    --eval_steps=1000 \
    --eval_strategy=no \
    --save_steps=1000 \
    --logging_strategy=steps \
    --logging_steps=1 \
    --bf16=true \
    --tf32=true \
    --ddp_timeout=1800000 \
    --use_liger_kernel=false \
    --gradient_checkpointing=true \
    --gradient_checkpointing_kwargs='{"use_reentrant": false}' \
    --group_by_length=false \
    --deepspeed=/root/workspace/ds_config/ZeRO_2_act_check.json