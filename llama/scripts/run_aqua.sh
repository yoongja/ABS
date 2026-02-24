CUDA_VISIBLE_DEVICES=3 python run.py \
    --backend llama-2-7b \
    --temperature 0.7 \
    --task aqua \
    --n_generate_sample 4 \
    --task_start_index 0 \
    --max_tokens 128 \
    --method_generate beam_sample \
    --method_evaluate value \
    --method_select beam_search \
    --n_select_sample 4 \
    --prompt_sample cot \
    --n_lambda_value 1 \
    --beam_adjustment True \
    ${@}