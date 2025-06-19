source ~/.bashrc
conda activate farms_imgcls  
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
src_path=$(pwd) 
ckpt_src_path=${src_path}/checkpoints/tempbalance
data_path=${src_path}/data/cv

for SLURM_ARRAY_TASK_ID in 5 8 9 12 13   18 21 22 25 26   32 35 36 39 40   47 50 51 54 55 
    do
        cfg=$(sed -n "$SLURM_ARRAY_TASK_ID"p ${src_path}/bash_scripts/config/farms_imagecls_main.txt)

        netType=$(echo $cfg | cut -f 1 -d ' ')
        width=$(echo $cfg | cut -f 2 -d ' ')
        depth=$(echo $cfg | cut -f 3 -d ' ')
        dataset=$(echo $cfg | cut -f 4 -d ' ')
        num_epochs=$(echo $cfg | cut -f 5 -d ' ')
        lr_sche=$(echo $cfg | cut -f 6 -d ' ')
        seed_lst=$(echo $cfg | cut -f 7 -d ' ')
        lr=$(echo $cfg | cut -f 8 -d ' ')
        weight_decay=$(echo $cfg | cut -f 9 -d ' ')
        pl_fitting=$(echo $cfg | cut -f 10 -d ' ')
        remove_first=$(echo $cfg | cut -f 11 -d ' ')
        remove_last=$(echo $cfg | cut -f 12 -d ' ')
        metric=$(echo $cfg | cut -f 13 -d ' ')
        assign_func=$(echo $cfg | cut -f 14 -d ' ')
        batchnorm=$(echo $cfg | cut -f 15 -d ' ')
        lr_min_ratio=$(echo $cfg | cut -f 16 -d ' ')
        lr_max_ratio=$(echo $cfg | cut -f 17 -d ' ')
        xmin_pos=$(echo $cfg | cut -f 18 -d ' ')
        sg=$(echo $cfg | cut -f 19 -d ' ')
        bn_type=$(echo $cfg | cut -f 20 -d ' ')
        use_tb=$(echo $cfg | cut -f 21 -d ' ')
        optim_type=$(echo $cfg | cut -f 22 -d ' ')
        ###FARMs
        use_sliding_window=$(echo $cfg | cut -f 23 -d ' ')
        num_row_samples=$(echo $cfg | cut -f 24 -d ' ')
        Q_ratio=$(echo $cfg | cut -f 25 -d ' ')
        step_size=$(echo $cfg | cut -f 26 -d ' ')
        sampling_ops_per_dim=$(echo $cfg | cut -f 27 -d ' ')
        batch_size=$(echo $cfg | cut -f 28 -d ' ')


        # saving folders of trained models 
        base_path=${ckpt_src_path}/${dataset}/TB_${use_tb}/${netType}_${depth}_${width}
        base_path=${base_path}/${metric}_plfitting_${pl_fitting}_xminpos${xmin_pos}_${assign_func}
        base_path=${base_path}/${optim_type}_min${lr_min_ratio}_max${lr_max_ratio}_init${lr}_${lr_sche}_bsz${batch_size}
        base_path=${base_path}/snr_sg${sg}/refirst${remove_first}_relast${remove_last}_bn${batchnorm}_type_${bn_type}
        base_path=${base_path}/slidingwindow_${use_sliding_window}_numrowsamples${num_row_samples}_Qratio${Q_ratio}_stepsize${step_size}_samplingopsperdim${sampling_ops_per_dim}

        array=($(echo $seed_lst | tr ',' ' '))

        # run three random seeds 
        for seed in 43 37 13 
            do 
                echo run experiment with seed $seed
                ckpt_folder=epochs${num_epochs}_wd${weight_decay}_seed${seed}
                mkdir -p ${base_path}/${ckpt_folder}

                CUDA_VISIBLE_DEVICES=4 python main_tb.py \
                    --lr ${lr} \
                    --net-type ${netType} \
                    --depth ${depth} \
                    --widen-factor ${width} \
                    --num-epochs ${num_epochs} \
                    --seed ${seed} \
                    --dataset ${dataset} \
                    --use-tb ${use_tb} \
                    --optim-type ${optim_type} \
                    --lr-sche ${lr_sche} \
                    --weight-decay ${weight_decay} \
                    --pl-fitting ${pl_fitting} \
                    --remove-last-layer ${remove_last} \
                    --remove-first-layer ${remove_first} \
                    --esd-metric-for-tb ${metric} \
                    --assign-func ${assign_func} \
                    --batchnorm ${batchnorm} \
                    --lr-min-ratio ${lr_min_ratio} \
                    --lr-max-ratio ${lr_max_ratio} \
                    --xmin-pos ${xmin_pos} \
                    --batchnorm-type ${bn_type} \
                    --sg ${sg} \
                    --ckpt-path ${base_path}/${ckpt_folder} \
                    --datadir ${data_path} \
                    --use-sliding-window ${use_sliding_window} \
                    --row-samples ${num_row_samples} \
                    --Q-ratio ${Q_ratio} \
                    --step-size ${step_size} \
                    --sampling-ops-per-dim ${sampling_ops_per_dim} \
                    --batch-size ${batch_size} 
            done
    done

# bash bash_scripts/tb_farms_imagecls_main.sh