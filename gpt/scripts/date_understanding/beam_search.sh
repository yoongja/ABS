python run_new.py \
    --task date_understanding \
    --n_generate_sample 4 \
    --max_tokens 128 \
    --task_start_index 0 \
    --method_generate et_sample \
    --method_evaluate value \
    --method_select beam_search \
    --n_select_sample 2 \
    --prompt_sample cot \
    --n_lambda_value 0.4 \
    --beam_adjustment false \
    ${@}