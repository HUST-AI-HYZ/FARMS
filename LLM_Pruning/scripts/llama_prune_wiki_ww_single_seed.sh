source ~/.bashrc
conda activate farms_prune_llm
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

src_path=$(pwd)

for SLURM_ARRAY_TASK_ID in {41..48..1}
    do
        cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p ${src_path}/scripts/llm_pruning_main.txt)
        ###parse config
        model=$(echo $cfg | cut -f 1 -d ' ')
        prune_method=$(echo $cfg | cut -f 2 -d ' ')
        sparsity_ratio=$(echo $cfg | cut -f 3 -d ' ')
        WW_metric=$(echo $cfg | cut -f 4 -d ' ')
        epsilon=$(echo $cfg | cut -f 5 -d ' ')
        result_save=$(echo $cfg | cut -f 6 -d ' ')
        num_row_samples=$(echo $cfg | cut -f 7 -d ' ')
        Q_ratio=$(echo $cfg | cut -f 8 -d ' ')
        step_size=$(echo $cfg | cut -f 9 -d ' ')
        sampling_ops_per_dim=$(echo $cfg | cut -f 10 -d ' ')
        use_sliding_window=$(echo $cfg | cut -f 11 -d ' ')
        WW_metric_cache=$(echo $cfg | cut -f 12 -d ' ')
        mapping_type=$(echo $cfg | cut -f 13 -d ' ')

        for seed in 0
            do
                CUDA_VISIBLE_DEVICES=7  python  main.py    \
                --model           $model            \
                --prune_method    $prune_method     \
                --sparsity_ratio  $sparsity_ratio   \
                --WW_metric       $WW_metric        \
                --epsilon         $epsilon          \
                --save            $result_save      \
                --num_row_samples $num_row_samples  \
                --Q_ratio         $Q_ratio          \
                --step_size       $step_size        \
                --sampling_ops_per_dim $sampling_ops_per_dim \
                --use_sliding_window $use_sliding_window \
                --WW_metric_cache $WW_metric_cache \
                --wikitext  \
                --seed $seed 
            done
    done
