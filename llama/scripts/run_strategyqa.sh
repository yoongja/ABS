CUDA_VISIBLE_DEVICES=0,1,2 python run.py \
    --backend llama-2-13b \
    --temperature 0.7 \
    --task strategyqa \
    --n_generate_sample 4 \
    --task_start_index 0 \
    --max_tokens 128 \
    --method_generate beam_sample \
    --method_evaluate value \
    --method_select beam_search \
    --n_select_sample 2 \
    --prompt_sample cot \
    --n_lambda_value 0.4 \
    --beam_adjustment True \
    ${@}