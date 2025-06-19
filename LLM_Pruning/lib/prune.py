import time
import heapq
import torch
import torch.nn as nn
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders 
import numpy as np
from collections import defaultdict

import weightwatcher as ww
from .esd_utils import net_esd_estimator
from .esd_utils_farms import net_esd_estimator_farms
import os

from .utils import get_weights, get_modules


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 


    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
    else:    
        layers = model.model.layers  
        
        
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 


def check_sparsity_mask(mask):
    W = mask
    count = 0 
    total_params = 0
    count += (W!=0).sum().item()
    total_params += W.numel()

    print(f" density {float(count)/total_params:.6f}")
    

def check_outlier(mask,threshold):
    W = mask
    count = 0 
    total_params = 0
    
    max_shred=torch.max(W)*threshold
    count += (W>max_shred).sum().item()
    total_params += W.numel()

    outlier_ratio=float(count)/total_params*100
    
    return outlier_ratio


def check_outlier_mean(mask,threshold):
    W = mask
    count = 0 
    total_params = 0
    
    max_shred=torch.mean(W)*threshold
    count += (W>max_shred).sum().item()
    total_params += W.numel()

    outlier_ratio=float(count)/total_params*100
    
    return outlier_ratio

def prepare_calibration_input(model, dataloader, device):
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 

    layers[0] = layers[0].module
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    return inps, outs, attention_mask, position_ids

def prepare_calibration_input_opt(model, dataloader, device):
    layers = model.model.decoder.layers
    
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None}
    
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
        
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
        
    layers[0] = layers[0].module
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    return inps, outs, attention_mask, None


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def owl_sparsity(args, model, tokenizer, device=torch.device("cuda:0")):

    all_layer_ratio=[]
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
            
    print ("inps",inps)
    if "opt" in args.model:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers
    
    for i in range(len(layers)):
        layer = layers[i]

        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
            
        layer_wmetric=[]

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            
            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            layer_wmetric.append(W_metric)    
                
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

        layer_wmetric = torch.cat([torch.flatten(x.cpu()) for x in layer_wmetric])
        
        for out_ratio in [args.Hyper_m]:
            out_ratio_layer=check_outlier_mean(layer_wmetric,out_ratio)
            print ("layer outlier ratio",out_ratio,out_ratio_layer)
        
        all_layer_ratio.append(out_ratio_layer)
    print ("before adjustment",all_layer_ratio)
    
    # mapping the OWL to layerwise pruning ratios
    all_layer_ratio=np.array(all_layer_ratio)
    all_layer_ratio = ((all_layer_ratio - all_layer_ratio.min()) * (1/(all_layer_ratio.max() - all_layer_ratio.min()) * args.Lamda*2))
    all_layer_ratio=all_layer_ratio-np.mean(all_layer_ratio)+(1-args.sparsity_ratio)
    print (all_layer_ratio,np.mean(all_layer_ratio),np.max(all_layer_ratio),np.min(all_layer_ratio))
    print ("after adjustment",all_layer_ratio)
    
    return all_layer_ratio

def ww_sparsity(args, model, device=torch.device("cuda:0"), s1=0.8, s2=1.2, ratios=None, prune_n=0, prune_m=0):
    if "opt" in args.model:
        blocks = model.model.decoder.layers    
    else:
        blocks = model.model.layers
    
    layers = [find_layers(blocks)]
    prunables = []
    for layer in layers:
        for name in layer:
            prunables.append(layer[name].weight.numel())

    layer_num_in_block = int(len(prunables) / len(blocks))

    
    
    if args.WW_metric in ['alpha_mid_tbv3', 'alpha_peak_farms']:
        # make fir 
        if not os.path.exists(f"{args.WW_metric_cache}/{args.WW_metric}"):
            os.makedirs(f"{args.WW_metric_cache}/{args.WW_metric}")
        # backup path
        backup_path = f"{args.WW_metric_cache}/{args.WW_metric}/usesld{args.use_sliding_window}_numrow{args.num_row_samples}_Q{args.Q_ratio}_step{args.step_size}_ops{args.sampling_ops_per_dim}.npy"
    else:
        if not os.path.exists(args.WW_metric_cache):
            os.makedirs(args.WW_metric_cache)
        backup_path = f"{args.WW_metric_cache}/{args.WW_metric}.npy"



    if os.path.exists(backup_path):
        metrics = np.load(backup_path)
        
        #if args.mapping_type == 'block_wise':
        #    block_metrics = [np.mean(metrics[i:i+layer_num_in_block]) for i in range(0, len(metrics), layer_num_in_block)]
        #    metrics = [i for i in block_metrics for j in range(layer_num_in_block)]
        
    else:
        print("Calculating WW metric by WeightWatcher")
        if args.WW_metric in ['alpha_mid_tbv3', 'alpha_peak_farms', 'alpha_mid', 'alpha_peak']:
            print("no ww")
        else:
            watcher = ww.WeightWatcher(model=blocks)
            details = watcher.analyze(mp_fit=True, randomize=True)
        print("weightwatcher analysis done")
        
        if args.WW_metric == 'entropy':
            metrics = np.array(details.entropy)
        elif args.WW_metric == 'alpha':
            metrics = np.array(details.alpha)
        elif args.WW_metric == 'alpha_mid':
            metrics = net_esd_estimator(blocks,
                fix_fingers='xmin_mid'
            )
            metrics = metrics['alpha']
        
        ##############################New code for WWsampling
        elif args.WW_metric == 'alpha_peak':
            metrics = net_esd_estimator(blocks,
                fix_fingers='xmin_peak'
            )
            np.save(f"{args.WW_metric_cache}/{args.WW_metric}_eigsinclude.npy", metrics)
            metrics = metrics['alpha']
        elif args.WW_metric == 'mp_softrank':
            metrics = np.array(details.mp_softrank)
        elif args.WW_metric == 'stable_rank':
            metrics = np.array(details.stable_rank)
        elif args.WW_metric == 'random_distance':
            metrics = np.array(details.rand_distance)
        elif args.WW_metric == 'log_norm':
            metrics = np.array(details.log_norm)
        elif args.WW_metric == 'log_spectral_norm':
            metrics = np.array(details.log_spectral_norm)
            
        elif args.WW_metric == 'alpha_weighted':
            metrics = np.array(details.alpha_weighted)
        elif args.WW_metric == 'log_alpha_norm':
            metrics = np.array(details.log_alpha_norm)

        elif args.WW_metric == 'spectral_norm':
            metrics = np.array(details.spectral_norm)
        
        ###New code for WWsampling
        elif args.WW_metric == 'alpha_peak_farms':
            metrics = net_esd_estimator_farms(blocks,
                fix_fingers='xmin_peak',
                use_sliding_window = args.use_sliding_window,
                num_row_samples = args.num_row_samples,
                Q_ratio = args.Q_ratio,
                step_size = args.step_size,
                sampling_ops_per_dim = args.sampling_ops_per_dim
            )
            np.save(f"{args.WW_metric_cache}/{args.WW_metric}/usesld{args.use_sliding_window}_numrow{args.num_row_samples}_Q{args.Q_ratio}_step{args.step_size}_ops{args.sampling_ops_per_dim}_eigsinclude.npy", metrics)
            metrics = metrics['alpha']
        else:
            raise NotImplementedError
            
            
            
            
        if args.WW_metric in ['alpha_peak_farms']:
            np.save(f"{args.WW_metric_cache}/{args.WW_metric}/usesld{args.use_sliding_window}_numrow{args.num_row_samples}_Q{args.Q_ratio}_step{args.step_size}_ops{args.sampling_ops_per_dim}.npy", metrics)
        elif args.WW_metric in ['alpha_peak']:
            np.save(f"{args.WW_metric_cache}/{args.WW_metric}.npy", metrics)
        else:
            raise NotImplementedError
        



    if args.mapping_type == 'block_wise':
        block_metrics = [np.mean(metrics[i:i+layer_num_in_block]) for i in range(0, len(metrics), layer_num_in_block)]
        metrics = [i for i in block_metrics for j in range(layer_num_in_block)]
        
        scores = torch.tensor(metrics)
        prunables = torch.tensor(prunables)

        # balance allocation
        max = torch.max(scores)
        min = torch.min(scores)
        
        layerwise_pruning_ratios = (((scores - min) / (max - min)) * (s2 - s1) + s1)
        scaler = torch.sum(prunables) * args.sparsity_ratio / (torch.sum(prunables * layerwise_pruning_ratios))  
        layerwise_pruning_ratios = layerwise_pruning_ratios * scaler
        layerwise_pruning_ratios = layerwise_pruning_ratios.cpu().numpy().tolist()
    elif args.mapping_type == 'mixed_wise':
        block_metrics = [np.mean(metrics[i:i+layer_num_in_block]) for i in range(0, len(metrics), layer_num_in_block)]
        ##################wait for code
        raise NotImplementedError
    elif args.mapping_type == 'pure_layer_wise':
        raise NotImplementedError
    
            
    
    print(metrics)
    print("Final layerwise_pruning_ratios")
    print(layerwise_pruning_ratios)
    ####we need to save the layer scores for each layer and visulaize the scores
    
    ###thresholding if ratio is greater than 1
    if args.sparsity_ratio != 0.7:
        for i in range(len(layerwise_pruning_ratios)):
            if layerwise_pruning_ratios[i]>1:
                layerwise_pruning_ratios[i] = 0.998
    else:
        for i in range(len(layerwise_pruning_ratios)):
            if layerwise_pruning_ratios[i]>1:
                layerwise_pruning_ratios[i] = 0.99999
    
    return layerwise_pruning_ratios

#########################################################################################################################

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, ratios=None):
    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
    else:    
        layers = model.model.layers
    
    layer_num = len(find_layers(layers))
    if ratios is None:
        ratios = [args.sparsity_ratio for i in range(layer_num)]

    k=0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            print(ratios[k])
            thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*ratios[k])].cpu()
            W_mask = (W_metric<=thresh)
            k+=1

            W[W_mask] = 0
    

def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, ratios=None):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)


    print ("inps",inps)

    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
    else:    
        layers = model.model.layers
    
    layer_num = len(find_layers(layers))
    if ratios is None:
        ratios = [args.sparsity_ratio for i in range(layer_num)]
    
    k=0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
            
        for j in range(args.nsamples):
            #print("sample",j)
            #print(f"pruning layer {layer}")
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*ratios[k])]
                    k+=1
                    W_mask.scatter_(1, indices, True)
#             print ("W_mask",W_mask)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, ratios=None):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
    else:    
        layers = model.model.layers        
    
    
    layer_num = len(find_layers(layers))
    if ratios is None:
        ratios = [args.sparsity_ratio for i in range(layer_num)]
    
    print('Ready.')
    k=0
    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            print("using !")
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            if "OPT" in model.__class__.__name__:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(ratios[k], prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()
            k+=1

        for j in range(args.nsamples):
            if "OPT" in model.__class__.__name__:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

###############################################################################################################
def prune_magnitude_ww(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    s1 = 1.0 - args.epsilon
    s2 = 1.0 + args.epsilon
    
    # calculate ww-based layer
    all_layer_ratio = ww_sparsity(args, model, device, s1, s2)
    
    # we should save the all_layer_ratio
    np.save(f"./all_layer_ratio/usesld{args.use_sliding_window}_numrow{args.num_row_samples}_Q{args.Q_ratio}_step{args.step_size}_ops{args.sampling_ops_per_dim}_sparsity{args.sparsity_ratio}_eps{args.epsilon}_layer_ratios.npy", all_layer_ratio)
    print("\n\n\n\n All layer ratio saved\n\n\n\n")
    
    # magnitude pruning
    prune_magnitude(args, model, tokenizer, device, ratios=all_layer_ratio)
    

def prune_wanda_ww(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    s1 = 1.0 - args.epsilon
    s2 = 1.0 + args.epsilon
    all_layer_ratio = ww_sparsity(args, model, device, s1, s2)

    # wanda pruning
    prune_wanda(args, model, tokenizer, device, ratios=all_layer_ratio)
    
def prune_sparsegpt_ww(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    s1 = 1.0 - args.epsilon
    s2 = 1.0 + args.epsilon
    all_layer_ratio = ww_sparsity(args, model, device, s1, s2)
    
    # sparsegpt pruning
    prune_sparsegpt(args, model, tokenizer, device, ratios=all_layer_ratio)
    

#############################################################################################################

def prune_magnitude_outlier(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    # calculate outlier ratio
    all_block_ratio = owl_sparsity(args, model, tokenizer, device)
    all_block_ratio = 1 - all_block_ratio
    
    if "opt" in args.model:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers
    
    block_len = len(find_layers(layers[0]))
    all_layer_ratio = [i for i in all_block_ratio for j in range(block_len)]
    
    # magnitude pruning
    prune_magnitude(args, model, tokenizer, device, ratios=all_layer_ratio)
    
def prune_wanda_outlier(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    # calculate outlier ratio
    all_block_ratio = owl_sparsity(args, model, tokenizer, device)
    all_block_ratio = 1 - all_block_ratio
    
    if "opt" in args.model:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers
    
    block_len = len(find_layers(layers[0]))
    all_layer_ratio = [i for i in all_block_ratio for j in range(block_len)]
    
    # wanda pruning
    prune_wanda(args, model, tokenizer, device, ratios=all_layer_ratio)

def prune_sparsegpt_outlier(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    # calculate outlier ratio
    all_block_ratio = owl_sparsity(args, model, tokenizer, device)
    all_block_ratio = 1 - all_block_ratio
    
    if "opt" in args.model:
        layers=model.model.decoder.layers
        
    else:
        layers = model.model.layers
    
    block_len = len(find_layers(layers[0]))
    all_layer_ratio = [i for i in all_block_ratio for j in range(block_len)]
    
    # sparsegpt pruning
    prune_sparsegpt(args, model, tokenizer, device, ratios=all_layer_ratio)


def prune_wanda_ww_structure(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, ratios=None):
    layerwise_n = ww_sparsity(args, model, device, prune_n=prune_n, prune_m=prune_m)
    
    ############## prune
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibdation data")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        if "OPT" in model.__class__.__name__:
            inps, outs, attention_mask, position_ids = prepare_calibration_input_opt(model, dataloader, device)
        else:
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    # print ("inps",inps)
    if "opt" in args.model:
        layers=model.model.decoder.layers
    else:
        layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        prune_n = int(layerwise_n[i])
        print('Layer {} prune_n {} prune_m {}'.format(i, prune_n, prune_m))

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            activation_data=torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)

            # print ("W_mask",W_mask)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                if "OPT" in model.__class__.__name__:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
                else:
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()