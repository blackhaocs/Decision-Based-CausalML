import torch
from Metric import * 

def get_per_capita_response_and_cost(t_hat, t, yr, yc, weights):

    matching_idx = torch.where(t_hat == t)[0]
    
    S = torch.bincount(t[matching_idx], minlength=2)
    S = torch.sum(S / weights)
    V = torch.sum(yr[matching_idx]) / S
    C = torch.sum(yc[matching_idx]) / S

    return V.cuda(), C.cuda(), S.cuda()

def recover_solution(r_hat, c_hat, lambda_):

    a = r_hat - lambda_ * c_hat
    t_hat = torch.argmax(a, dim=1)

    return t_hat

def improved_finite_difference(t, yr, yc, r_hat, c_hat, lambda_, weights, h, clip=None):
        
    
    yr = yr.float().squeeze()
    yc = yc.float().squeeze()
    weights = weights.float().cuda()
    t = t.squeeze().to(torch.int64)

    dDDL_dr = torch.zeros_like(r_hat).cuda()
    dDDL_dc = torch.zeros_like(c_hat).cuda()

    #Save the result and use it to calculate the perturbation.
    a = r_hat - lambda_ * c_hat 
    #The allocation policy under lambda_ is computed based on r_hat and c_hat.
    t_hat = recover_solution(r_hat, c_hat, lambda_) 

    #Calculate the per capita benefit, per capita cost, and number of matched samples under lambda_.
    r_bar_1, c_bar_1, N = get_per_capita_response_and_cost(t_hat, t, yr, yc, weights) 

    #Store the index of the successfully matched samples.
    mismatching_idx = torch.where(t != t_hat)[0] 
    #Store the index of the unsuccessfully matched samples.
    matching_idx = torch.where(t == t_hat)[0]

    #Extend N to an (nBatch) tensor.
    N = N.expand(r_hat.shape[0]).float().cuda()

    #The perturbation matrix with respect to r: a (nBatch, ntreatment) tensor
    p_r = torch.zeros_like(r_hat).float().cuda() 
    #The perturbation matrix with respect to r: a (nBatch, ntreatment) tensor
    p_c = torch.zeros_like(c_hat).float().cuda() 
    #Extend the pre-perturbation per capita revenue to a (nBatch) tensor.
    r_bar_1 = r_bar_1.expand(r_hat.shape[0]).float().cuda() 
    #Extend the pre-perturbation per capita cost to a (nBatch) tensor.
    c_bar_1 = c_bar_1.expand(c_hat.shape[0]).float().cuda() 

    #Calculate pre-perturbation dual decision loss.
    DDL_1 = r_bar_1 - lambda_  * c_bar_1 
    
    #Declare post-perturbation per capita revenue: a (nBatch) tensor
    r_bar_2 = torch.zeros(r_hat.shape[0]).float().cuda() 
    #Declare post-perturbation per capita cost: a (nBatch) tensor
    c_bar_2 = torch.zeros(c_hat.shape[0]).float().cuda() 
    #Declare post-perturbation dual decision loss: a (nBatch) tensor
    DDL_2 = torch.zeros_like(DDL_1).float().cuda() 

    #Calculate the perturbation of mismatching samples.
    p_r[mismatching_idx, t[mismatching_idx]] = a[mismatching_idx, t_hat[mismatching_idx]] - a[mismatching_idx, t[mismatching_idx]] + 1e-6
    p_c[mismatching_idx, t[mismatching_idx]] = (a[mismatching_idx,t[mismatching_idx]] - a[mismatching_idx, t_hat[mismatching_idx]]) / lambda_ - 1e-6

    p_r[mismatching_idx, t_hat[mismatching_idx]] = -p_r[mismatching_idx, t[mismatching_idx]]
    p_c[mismatching_idx, t_hat[mismatching_idx]] = -p_c[mismatching_idx, t[mismatching_idx]]


    #Calculate the per capita revenue and per capita cost after the mismatching samples are perturbed.
    r_bar_2[mismatching_idx] = (r_bar_1[mismatching_idx] * N[mismatching_idx] + yr[mismatching_idx]) / (N[mismatching_idx] + 1 / weights[t[mismatching_idx]])
    c_bar_2[mismatching_idx] = (c_bar_1[mismatching_idx] * N[mismatching_idx] + yc[mismatching_idx]) / (N[mismatching_idx] + 1 / weights[t[mismatching_idx]])
       


    DDL_2[mismatching_idx] = r_bar_2[mismatching_idx] - lambda_ * c_bar_2[mismatching_idx]


    values, indices = torch.topk(a, 2, dim=1)  
    idx1 = indices[:, 1] 



    #Calculate the perturbation of matching samples.
    p_r[matching_idx, t[matching_idx]] = a[matching_idx, idx1[matching_idx]] - a[matching_idx, t[matching_idx]] - 1e-6
    p_c[matching_idx, t[matching_idx]] = (a[matching_idx, t[matching_idx]] - a[matching_idx, idx1[matching_idx]]) / lambda_ + 1e-6

    p_r[matching_idx, idx1[matching_idx]] = -p_r[matching_idx, t[matching_idx]]
    p_c[matching_idx, idx1[matching_idx]] = -p_c[matching_idx, t[matching_idx]]


    #Calculate the per capita revenue and per capita cost after the matching samples are perturbed.
    r_bar_2[matching_idx] = (r_bar_1[matching_idx] * N[matching_idx] - yr[matching_idx]) / (N[matching_idx] - 1 / weights[t[matching_idx]])
    c_bar_2[matching_idx] = (c_bar_1[matching_idx] * N[matching_idx] - yc[matching_idx]) / (N[matching_idx] - 1 / weights[t[matching_idx]])
        

    DDL_2[matching_idx] = r_bar_2[matching_idx] - lambda_ * c_bar_2[matching_idx] 

    if clip == 'fix':        
        mask_r = (torch.abs(p_r) < h) & (p_r != 0)
        p_r[mask_r] = torch.where(p_r[mask_r] > 0, torch.tensor(h).cuda(), torch.tensor(-h).cuda())      
        mask_c = (torch.abs(p_c) < h) & (p_c != 0)
        p_c[mask_c] = torch.where(p_c[mask_c] > 0, torch.tensor(h).cuda(), torch.tensor(-h).cuda())
    elif clip == 'max':
        max_pr = torch.max(torch.abs(p_r))
        max_pc = torch.max(torch.abs(p_c))
        p_r = torch.where(p_r > 0, max_pr, torch.where(p_r < 0, -max_pr, p_r))
        p_c = torch.where(p_c > 0, max_pc, torch.where(p_c < 0, -max_pc, p_c))
    
    #Calculate the gradient of the mismatching samples
    dDDL_dr[mismatching_idx, t[mismatching_idx]] = (DDL_2[mismatching_idx] - DDL_1[mismatching_idx]) / p_r[mismatching_idx, t[mismatching_idx]]
    dDDL_dc[mismatching_idx, t[mismatching_idx]] = (DDL_2[mismatching_idx] - DDL_1[mismatching_idx]) / p_c[mismatching_idx, t[mismatching_idx]]
    dDDL_dr[mismatching_idx, t_hat[mismatching_idx]] = (DDL_2[mismatching_idx] - DDL_1[mismatching_idx]) / p_r[mismatching_idx, t_hat[mismatching_idx]]
    dDDL_dc[mismatching_idx, t_hat[mismatching_idx]] = (DDL_2[mismatching_idx] - DDL_1[mismatching_idx]) / p_c[mismatching_idx, t_hat[mismatching_idx]]


    #Calculate the gradient of the matching samples
    dDDL_dr[matching_idx, t[matching_idx]] =(DDL_2[matching_idx] - DDL_1[matching_idx]) / p_r[matching_idx, t[matching_idx]]
    dDDL_dc[matching_idx, t[matching_idx]] = (DDL_2[matching_idx] - DDL_1[matching_idx]) / p_c[matching_idx, t[matching_idx]]
    dDDL_dr[matching_idx, idx1[matching_idx]] = (DDL_2[matching_idx] - DDL_1[matching_idx]) / p_r[matching_idx, idx1[matching_idx]]
    dDDL_dc[matching_idx, idx1[matching_idx]] = (DDL_2[matching_idx] - DDL_1[matching_idx]) / p_c[matching_idx, idx1[matching_idx]]

    if clip == 'fix':
        dDDL_dr[~mask_r] = 0
        dDDL_dc[~mask_c] = 0 

    return dDDL_dr, dDDL_dc