import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, LlamaTokenizer, LlamaForCausalLM
from collections import defaultdict

from lib.esd_utils_farms import net_esd_estimator_farms
from lib.esd_utils import net_esd_estimator
from lib.prune import prune_wanda, prune_sparsegpt, prune_magnitude, prune_wanda_ww, prune_sparsegpt_ww, prune_magnitude_ww, check_sparsity 
from lib.eval import eval_ppl, eval_zero_shot, eval_zero_shot_new
import sys
import random
import datasets


def get_llm(model, cache_dir="llm_weights"):
    
    if model == 'meta-llama/Llama-3.1-8B':
        def skip(*args, **kwargs):
            pass
        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip
        model = LlamaForCausalLM.from_pretrained(model, torch_dtype="auto", low_cpu_mem_usage=True, cache_dir = cache_dir, device_map = "auto")
        model.seqlen = 2048
        
        return model
    else:
        model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype = torch.float16,
        cache_dir = cache_dir,
        low_cpu_mem_usage=True,
        device_map = "auto",
    )
        model.seqlen = 2048
    
        return model
    
    
    


def main():
    parser = argparse.ArgumentParser()  
    parser.add_argument('--model', type=str, help="model type")
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument('--prune_method', type=str)
    parser.add_argument('--sparsity_type', type=str, default="unstructured", help='Structured pruning for N:M')
    parser.add_argument('--cache_dir', default="llm_weights", type=str)
    parser.add_argument('--save', type=str, default=None, help='Path to save results')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--mapping_type', type=str, default="block_wise", help='Type of mapping for structured pruning')
    # params for WW
    parser.add_argument("--save_pruned_metrics", action="store_true", default=False, help="save the pruned metrics")
    parser.add_argument("--WW_metric", default="alpha_peak", type=str, help="the WW-based metric to ues.")
    parser.add_argument("--WW_metric_cache", default="./metric_cache/llama-7b-hf")
    parser.add_argument("--epsilon", default=0.2, type=float, help="for pruning ratio allocation.")
    # evaluation benchmark
    parser.add_argument("--eval_zero_shot", action="store_true", help="evaluation on zero-shot tasks.")
    parser.add_argument("--wikitext", action="store_true", help="evaluation on wikitext.")
    # WWSampling
    parser.add_argument('--use_sliding_window', default=False, type=lambda x: (str(x).lower() == 'true'), help='use sliding window')
    parser.add_argument('--num_row_samples', type=int, default=100)
    parser.add_argument('--Q_ratio', type=float, default=2.0)
    parser.add_argument('--step_size', type=int, default=10)
    parser.add_argument('--sampling_ops_per_dim', type=int, default=10)
    args = parser.parse_args()
    
    print(args)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    
    model_name = args.model.split("/")[-1]
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    
    # print the matrix size of the model
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            print(name, m.weight.shape)
            
            if args.use_sliding_window:
                # Sanity check
                rows, cols = m.weight.shape
                num_row_samples = args.num_row_samples
                num_col_samples = int(num_row_samples * args.Q_ratio)
                
                # Num_row_samples Threshold            
                if args.num_row_samples >  min(rows, cols):
                    num_row_samples = min(args.num_row_samples, min(rows, cols))
                    num_col_samples = int(num_row_samples * args.Q_ratio)
                    print(f"Number of cow/row samples: {num_col_samples} x {num_row_samples}")


                # sampling_ops_per_dim cannot be too low
                if args.sampling_ops_per_dim < int(max(rows, cols) / min(num_row_samples, num_col_samples)):
                    sampling_ops_per_dim = max(int(max(rows, cols) / min(num_row_samples, num_col_samples)), args.sampling_ops_per_dim)
                    print(f"Sampling operations per dimension: {sampling_ops_per_dim}")
            else:
                pass
    print("model loaded")
    
    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))
    
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model or "70b" in args.model in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)
    

    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

        ################################################
        
        elif args.prune_method == "wanda_ww":
            prune_wanda_ww(args, model, tokenizer, device)

        elif args.prune_method == "magnitude_ww":
            prune_magnitude_ww(args, model, tokenizer, device)

        elif args.prune_method == "sparsegpt_ww":
            prune_sparsegpt_ww(args, model, tokenizer, device)
            
    sparsity_ratio = check_sparsity(model)
    
    folder_path = os.path.join(args.save, f"WW_metric{args.WW_metric}_usesld{args.use_sliding_window}_numrow{args.num_row_samples}_Q{args.Q_ratio}_step{args.step_size}_ops{args.sampling_ops_per_dim}/epsilon{args.epsilon}")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    if args.wikitext:
        ppl_test = eval_ppl(args, model, tokenizer, device)
        print(f"wikitext perplexity {ppl_test}")

        save_filepath = os.path.join(folder_path, f"perplexity_{args.prune_method}_sparsity_{args.sparsity_ratio}_seed{args.seed}.txt")
        with open(save_filepath, "w") as f:
            print("method\tactual_sparsity\tppl_test", file=f, flush=True)
            print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)
    
    # zero-shot tasks evaluation
    if args.eval_zero_shot:
        accelerate=False
        
        if "30b" in args.model or "65b" in args.model or "70b" in args.model or 'Llama-2-13b-hf' in args.model:
            accelerate=True

        task_list = ["boolq", "rte", "hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"] #
        num_shot = 0
        results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
        
        #max_samples = 20
        #results = eval_zero_shot_new(model_name, model, tokenizer, max_samples, task_list, accelerate)
        print("zero_shot evaluation results")
        print(results)

        save_filepath = os.path.join(folder_path, f"zero_shot_{args.prune_method}_sparsity_{args.sparsity_ratio}.txt")
        with open(save_filepath, "w") as f:
            print(f"{args.prune_method}:\n{results}", file=f, flush=True)
            #save the results dicts in txt
            #import json
            #json.dump(results, f, indent=4)    
            
    
    # save the ESD of the pruned model
    if args.save_pruned_metrics:
        if args.WW_metric == 'alpha_peak':
            metrics = net_esd_estimator(model,
                fix_fingers='xmin_peak'
            )
            metrics = metrics['alpha']
            np.save(f"{folder_path}/{args.prune_method}_{args.WW_metric}_{args.sparsity_ratio}_pruned.npy", metrics)
        elif args.WW_metric == 'alpha_peak_farms':
            metrics = net_esd_estimator_farms(model,
                fix_fingers='xmin_peak',
                use_sliding_window = args.use_sliding_window,
                num_row_samples = args.num_row_samples,
                Q_ratio = args.Q_ratio,
                step_size = args.step_size,
                sampling_ops_per_dim = args.sampling_ops_per_dim
            )
            metrics = metrics['alpha']
            np.save(f"{folder_path}/{args.prune_method}_{args.WW_metric}_usesld{args.use_sliding_window}_numrow{args.num_row_samples}_Q{args.Q_ratio}_step{args.step_size}_ops{args.sampling_ops_per_dim}_{args.sparsity_ratio}_pruned.npy", metrics)
        else:
            raise NotImplementedError(f"metric {args.WW_metric} is not implemented.")
    else:
        print("no metrics saved.")
        
    
    
    # save model if needed.    
    if args.save_model:
        save_model_path = os.path.join(args.save_model, f"{args.prune_method}_{args.sparsity_ratio}")
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        model.save_pretrained(save_model_path)
        tokenizer.save_pretrained(save_model_path)
    
    
if __name__ == '__main__':
    main()
