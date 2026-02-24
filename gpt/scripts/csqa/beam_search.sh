python run.py \
    --task commonsenseqa \
    --n_generate_sample 4 \
    --task_start_index 0 \
    --task_end_index -1 \
    --max_tokens 256 \
    --method_generate et_sample \
    --method_evaluate value \
    --method_select beam_search \
    --n_evaluate_sample 1 \
    --n_select_sample 2 \
    --n_lambda_value 1 \
    --prompt_sample cot \
    --beam_adjustment True \
    ${@}