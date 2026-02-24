python run_new.py \
    --task game24 \
    --backend gpt-4 \
    --n_generate_sample 4 \
    --task_start_index 0 \
    --task_end_index 100 \
    --max_tokens 256 \
    --method_generate et_sample \
    --method_evaluate value \
    --method_select beam_search \
    --n_select_sample 2 \
    --prompt_sample cot \
    --beam_adjustment True \
    ${@}
