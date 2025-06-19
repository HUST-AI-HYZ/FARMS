import torch
import torch.nn as nn
from operator import itemgetter
import numpy as np
import math
import tqdm
import re
import os
from .sampling import *
import time



def net_esd_estimator_farms(
        net = None,
        EVALS_THRESH = 0.00001,
        bins=100,
        fix_fingers='xmin_mid',
        xmin_pos = 2,
        conv_norm = 0.5,
        eigs_num_thresh=5,
        filter_zeros = False,
        # Sliding window sampling
        use_sliding_window = True,
        num_row_samples = 100,
        Q_ratio = 2.0, 
        step_size = 10,
        sampling_ops_per_dim = 10,
        waived_layers = [] ):
        
    
    results = {
    'alpha':[],
    'spectral_norm': [],
    'D': [],
    'longname':[],
    'eigs':[],
    'norm':[],
    'alpha_hat':[],
    'stable_rank':[],
    'norm_stable_rank':[],
    }
    

    print("======================================")
    print(f"fix_fingers: {fix_fingers}, xmin_pos: {xmin_pos}, conv_norm: {conv_norm}, filter_zeros: {filter_zeros}")
    print(f"use_sliding_window: {use_sliding_window}, num_row_samples: {num_row_samples}, Q_ratio: {Q_ratio}, step_size: {step_size}, sampling_ops_per_dim: {sampling_ops_per_dim}")
    print("======================================")

    device = next(net.parameters()).device  # type: ignore
    for name, m in net.named_modules(): # type: ignore    
        if name in waived_layers:
                continue
        
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) :    
            if use_sliding_window:
                ######### CNN Layers ##########
                if isinstance(m, nn.Conv2d):
                    matrix = m.weight.data.clone().to(device)
                    matrix = matrix.float()
                    Wmats, N, M, rf = conv2D_Wmats(matrix, channels=CHANNELS.UNKNOWN)
                    rows, cols = Wmats[0].shape
                    results['longname'].append(name)  
                      
                    if Q_ratio > 1:
                        num_row_samples_cnn = cols // Q_ratio
                        num_col_samples_cnn = cols
                    else:
                        num_row_samples_cnn = cols
                        num_col_samples_cnn = cols // (1 / Q_ratio)
                    num_row_samples_cnn = max(1, int(num_row_samples_cnn))
                    num_col_samples_cnn = max(1, int(num_col_samples_cnn))
    
                    division = int(rows // cols)
                    temp_results = {
                                'alpha':[],
                                'spectral_norm': [],
                                'D': [],
                                'longname':[],
                                'eigs':[],
                                'norm':[],
                                'alpha_hat':[],
                                'stable_rank':[],
                                'norm_stable_rank':[],
                                }
                    for i in range(division):
                        division_eigs = []
                        for W in Wmats:
                            submatrix = W[i*num_row_samples_cnn:(i+1)*num_row_samples_cnn, :num_col_samples_cnn]
                            submatrix *= math.sqrt(conv_norm)
                            division_eigs.append(torch.square(torch.linalg.svdvals(submatrix)))
                        
                        division_eigs = torch.cat(division_eigs)
                        division_eigs = torch.sort(division_eigs, descending=False).values
                        temp_results = analysis_esd(
                                results=temp_results,
                                eigs=division_eigs,
                                EVALS_THRESH=EVALS_THRESH,
                                bins=bins,
                                fix_fingers=fix_fingers,
                                xmin_pos=xmin_pos,
                                filter_zeros=filter_zeros,
                                device=device)
                            
                    results['alpha'].append(np.mean(temp_results['alpha']))
                    results['spectral_norm'].append(np.mean(temp_results['spectral_norm']))
                    results['D'].append(np.mean(temp_results['D']))
                    results['norm'].append(np.mean(temp_results['norm']))
                    results['alpha_hat'].append(np.mean(temp_results['alpha_hat']))
                    results['stable_rank'].append(np.mean(temp_results['stable_rank']))
                    results['norm_stable_rank'].append(np.mean(temp_results['norm_stable_rank']))
                    results['eigs'].append(np.concatenate(temp_results['eigs']).cpu().numpy())
                ######### Linear Layers ##########    
                elif isinstance(m, nn.Linear):
                    matrix = m.weight.data.clone().to(device)
                    matrix = matrix.float()
                    eigs = sampled_eigs(
                        matrix=matrix, isconv2d=isinstance(m, nn.Conv2d),
                        conv_norm=conv_norm, num_row_samples=num_row_samples,
                        Q_ratio=Q_ratio, step_size=step_size,
                        sampling_ops_per_dim=sampling_ops_per_dim
                    )
                    
                    if len(eigs) < eigs_num_thresh:
                        continue
                    else:
                        results['longname'].append(name)
                        results = analysis_esd(
                            results=results,
                            eigs=eigs,
                            EVALS_THRESH=EVALS_THRESH,
                            bins=bins,
                            fix_fingers=fix_fingers,
                            xmin_pos=xmin_pos,
                            filter_zeros=filter_zeros,
                            device=device
                        )

            elif not use_sliding_window:
                matrix = m.weight.data.clone().to(device)
                matrix = matrix.float()
                if isinstance(m, nn.Conv2d):
                    matrix = torch.flatten(matrix, start_dim=2) * math.sqrt(conv_norm)
                    matrix = matrix.transpose(1, 2).transpose(0, 1)
                eigs = torch.square(torch.linalg.svdvals(matrix).flatten())

                if len(eigs) < eigs_num_thresh:
                    continue
                else:
                    results['longname'].append(name)
                    results = analysis_esd(
                        results=results,
                        eigs=eigs,
                        EVALS_THRESH=EVALS_THRESH,
                        bins=bins,
                        fix_fingers=fix_fingers,
                        xmin_pos=xmin_pos,
                        filter_zeros=filter_zeros,
                        device=device
                    )
                
    return results







def analysis_esd(
        results,
        eigs,
        EVALS_THRESH=0.00001,
        bins=100,
        fix_fingers=None,
        xmin_pos=2,
        filter_zeros=False,
        device='cuda'): 
     
    if not isinstance(eigs, torch.Tensor):
        eigs = torch.tensor(eigs, device=device)
    
    #print("eigs_sum", eigs)     
    eigs_sum = torch.sum(eigs)
    max_eigs = torch.max(eigs)
    stable_rank = eigs_sum / max_eigs
    norm_stable_rank = eigs_sum / len(eigs)
    eigs = torch.sort(eigs).values
    spectral_norm = eigs[-1].item()
    fnorm = torch.sum(eigs).item()
    
    
    # Filter based on threshold
    nz_eigs = eigs[eigs > EVALS_THRESH] if filter_zeros else eigs
    if len(nz_eigs) == 0:
        nz_eigs = eigs
    N = len(nz_eigs)
    log_nz_eigs = torch.log(nz_eigs)



    # Alpha and D calculations (from before)
    if fix_fingers == 'xmin_mid':
        i = int(len(nz_eigs) / xmin_pos) 
        #i = N // xmin_pos
        xmin = nz_eigs[i]
        n = float(N - i)
        seq = torch.arange(n, device=device)
        final_alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
        final_D = torch.max(torch.abs(1 - (nz_eigs[i:] / xmin) ** (-final_alpha + 1) - seq / n))
    else:
        alphas = torch.zeros(N-1, device=device)
        Ds = torch.ones(N-1, device=device)
        if fix_fingers == 'xmin_peak':
            hist_nz_eigs = torch.log10(nz_eigs)
            min_e, max_e = hist_nz_eigs.min(), hist_nz_eigs.max()
            counts = torch.histc(hist_nz_eigs, bins, min=min_e, max=max_e) # type: ignore
            boundaries = torch.linspace(min_e, max_e, bins + 1) # type: ignore
            ih = torch.argmax(counts)
            xmin2 = 10 ** boundaries[ih]
            xmin_min = torch.log10(0.95 * xmin2)
            xmin_max = 1.5 * xmin2

        for i, xmin in enumerate(nz_eigs[:-1]):
            if fix_fingers == 'xmin_peak':
                if xmin < xmin_min:
                    continue
                if xmin > xmin_max:
                    break
            n = float(N - i)
            seq = torch.arange(n, device=device)
            alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
            alphas[i] = alpha
            if alpha > 1:
                Ds[i] = torch.max(torch.abs(1 - (nz_eigs[i:] / xmin) ** (-alpha + 1) - seq / n))

        min_D_index = torch.argmin(Ds)
        final_alpha = alphas[min_D_index]
        final_D = Ds[min_D_index]

    # Store results
    final_alpha = final_alpha.item()  # type: ignore
    final_D = final_D.item()  # type: ignore
    final_alphahat = final_alpha * math.log10(spectral_norm)

    results['alpha'].append(final_alpha)
    results['spectral_norm'].append(spectral_norm)
    results['D'].append(final_D)
    results['eigs'].append(nz_eigs.cpu().numpy())
    results['norm'].append(fnorm)
    results['alpha_hat'].append(final_alphahat)
    results['stable_rank'].append(stable_rank.item())
    results['norm_stable_rank'].append(norm_stable_rank.item())
        
    return results         
                
                


