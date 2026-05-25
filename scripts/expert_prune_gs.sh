
export PYTHONPATH=$PWD:$PYTHONPATH
# export HF_ENDPOINT=https://hf-mirror.com

calib_size=32

model_id="OLMoE"

python step/prune.py --model_name_or_path allenai/OLMoE-1B-7B-0125 $data_config \
    --expert_prune --preserve_n_experts 48 --expert_ranking_scope model \
    --expert_prune_metric routing_score \
    --max_steps $calib_size \
