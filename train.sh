python train.py \
    --model_name_or_path 'bert' \
    --max_length 32 \
    --trust_remote_code True \
    --epochs 4 \
    --train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --lr 3e-4 \
    --lr_warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --seed 42 \
    --eval_batch_size 16 \
    --eval_ratio 0.01 \
    --eval_interval 100 \
    --output_dir_name C:\Users\lijiahao\PycharmProjects\NLP_assignment2