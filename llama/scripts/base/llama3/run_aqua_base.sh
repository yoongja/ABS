CUDA_VISIBLE_DEVICES=0 python run_8b_base_beam.py \
    --backend llama-3.1-8b \
    --temperature 0.7 \
    --task aqua \
    --n_generate_sample 4 \
    --task_start_index 0 \
    --max_tokens 128 \
    --method_generate beam_sample \
    --method_evaluate value \
    --method_select base_beam \
    --n_select_sample 2 \
    --prompt_sample cot \
    --n_lambda_value 0.9 \
    --beam_adjustment False\
    ${@}