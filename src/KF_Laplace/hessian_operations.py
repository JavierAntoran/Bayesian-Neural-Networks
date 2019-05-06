from __future__ import division

import torch
# from src.base_net import *

def softmax_CE_preact_hessian(last_layer_acts):
    side = last_layer_acts.shape[1]
    I = torch.eye(side).type(torch.ByteTensor)
    # for i != j    H = -ai * aj -- Note that these are activations not pre-activations
    Hl = - last_layer_acts.unsqueeze(1) * last_layer_acts.unsqueeze(2)
    # for i == j    H = ai * (1 - ai)
    Hl[:, I] = last_layer_acts * (1 - last_layer_acts)
    return Hl


def layer_act_hessian_recurse(prev_hessian, prev_weights, layer_pre_acts):
    newside = layer_pre_acts.shape[1]
    batch_size = layer_pre_acts.shape[0]
    I = torch.eye(newside).type(torch.ByteTensor)  # .unsqueeze(0).expand([batch_size, -1, -1])

    #     print(d_act(layer_pre_acts).unsqueeze(1).shape, I.shape)
    B = prev_weights.data.new(batch_size, newside, newside).fill_(0)
    B[:, I] = (layer_pre_acts > 0).type(B.type())  # d_act(layer_pre_acts)
    D = prev_weights.data.new(batch_size, newside, newside).fill_(0)  # is just 0 for a piecewise linear
    #     D[:, I] = dd_act(layer_pre_acts) * act_grads

    Hl = torch.bmm(torch.t(prev_weights).unsqueeze(0).expand([batch_size, -1, -1]), prev_hessian)
    Hl = torch.bmm(Hl, prev_weights.unsqueeze(0).expand([batch_size, -1, -1]))
    Hl = torch.bmm(B, Hl)
    Hl = torch.matmul(Hl, B)
    Hl = Hl + D

    return Hl


def chol_scale_invert_kron_factor(factor, prior_scale, data_scale, upper=False):
    scaled_factor = data_scale * factor + prior_scale * torch.eye(factor.shape[0]).type(factor.type())
    inv_factor = torch.inverse(scaled_factor)
    chol_inv_factor = torch.cholesky(inv_factor, upper=upper)
    return chol_inv_factor


def sample_K_laplace_MN(MAP, upper_Qinv, lower_HHinv):
    # H = Qi (kron) HHi
    # sample isotropic unit variance mtrix normal
    Z = MAP.data.new(MAP.size()).normal_(mean=0, std=1)
    # AAT = HHi
    #     A = torch.cholesky(HHinv, upper=False)
    # BTB = Qi
    #     B = torch.cholesky(Qinv, upper=True)
    all_mtx_sample = MAP + torch.matmul(torch.matmul(lower_HHinv, Z), upper_Qinv)

    weight_mtx_sample = all_mtx_sample[:, :-1]
    bias_mtx_sample = all_mtx_sample[:, -1]

    return weight_mtx_sample, bias_mtx_sample