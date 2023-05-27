import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
from gcc_phat import gcc_phat
# Order = (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)

def quad(sin, cos):
    if sin >= 0:
        if cos >= 0:
            quadrant = 1
        else:
            quadrant = 2
    else:
        if cos>= 0:
            quadrant = 4
        else:
            quadrant = 3
    return quadrant


def lstsq_doa(dist_diff, verbose=False):
    A = torch.tensor([
        [torch.sin(torch.pi / torch.tensor(4.0)), -torch.cos(torch.pi / torch.tensor(4.0))],
        [torch.sin(torch.pi / torch.tensor(2.0)), -torch.cos(torch.pi / torch.tensor(2.0))],
        [torch.sin(torch.pi * torch.tensor(3.0) / torch.tensor(4.0)),
         -torch.cos(torch.pi * torch.tensor(3.0) / torch.tensor(4.0))],
        [torch.sin(torch.pi * torch.tensor(3.0) / torch.tensor(4.0)),
         -torch.cos(torch.pi * torch.tensor(3.0) / torch.tensor(4.0))],
        [torch.sin(torch.pi / torch.tensor(1.0)), -torch.cos(torch.pi / torch.tensor(1.0))],
        [torch.sin(torch.pi * torch.tensor(5.0) / torch.tensor(4.0)),
         -torch.cos(torch.pi * torch.tensor(5.0) / torch.tensor(4.0))],
    ])

    # print("A before distance", A)
    A = A * torch.tensor([
        [4.57],
        [4.57 * torch.sqrt(torch.tensor(2.0))],
        [4.57],
        [4.57],
        [4.57 * torch.sqrt(torch.tensor(2.0))],
        [4.57]
    ])

    dist_diff = dist_diff * 343.26/160.0
    print("dist_diff", dist_diff)
    print("A", A)
    results = torch.linalg.lstsq(A, dist_diff)
    if verbose:
        print(results[0])
    cos, sin = results[0]
    asin = torch.arcsin(sin)/torch.pi*180.0
    acos = torch.arccos(cos)/torch.pi*180.0
    if verbose:
        print("sin, cos", sin, cos)
        print("asin, acos", asin, acos)
    quadrant = quad(sin, cos)
    if verbose:
        print(quadrant)
    if quadrant == 1:
        angle = torch.tensor([asin, acos])
    elif quadrant == 2:
        angle = torch.tensor([180.0 - asin, acos])
    elif quadrant == 3:
        angle = torch.tensor([180.0 - asin, 360.0 - acos])
    else:
        angle = torch.tensor([360.0 + asin, 360.0 - acos])
    if verbose:
        print(angle)
    if torch.isnan(angle[0]):
        if torch.isnan(angle[1]):
            estimate = torch.tensor(0)
        else:
            estimate = angle[1]
    elif torch.isnan(angle[1]):
        estimate = angle[0]
    else:
        estimate = (angle[0] + angle[1])/2
    if verbose:
        print("estimate", estimate)
    return torch.tensor(estimate), torch.isnan(angle[0]) or torch.isnan(angle[1]), torch.isnan(angle[0]) and torch.isnan(angle[1])


def f(distance_pairs, theta):
    total = 0.0
    # Order = (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)

    total += torch.abs(distance_pairs[0] - (4.57 * torch.sin((torch.pi/torch.tensor(4.0)) - theta)))
    total += torch.abs(distance_pairs[1] - (4.57 * torch.sqrt(torch.tensor(2.0)) * torch.sin((torch.pi/torch.tensor(2.0)) - theta)))
    total += torch.abs(distance_pairs[2] - (4.57 * torch.sin((torch.pi*torch.tensor(3.0)/torch.tensor(4.0)) - theta)))
    total += torch.abs(distance_pairs[3] - (4.57 * torch.sin((torch.pi*torch.tensor(3.0)/torch.tensor(4.0)) - theta)))
    total += torch.abs(distance_pairs[4] - (4.57 * torch.sqrt(torch.tensor(2.0)) * torch.sin((torch.pi/torch.tensor(1.0)) - theta)))
    total += torch.abs(distance_pairs[5] - (4.57 * torch.sin((torch.pi*torch.tensor(5.0)/torch.tensor(4.0)) - theta)))
    return total


def try_all_doa(dist_diff):
    dist_diff = dist_diff * 343.26/160
    best_angle = 0
    lowest_val = torch.inf
    vals = []
    for i in range(360):
        angle_val = f(dist_diff, i/180*torch.pi)
        vals.append(angle_val)
        if angle_val < lowest_val:
            best_angle = i
            lowest_val = angle_val
    return best_angle, vals


def converge_doa(dist_diff):
    dist_diff = dist_diff * 343.26/160
    x=nn.Parameter(torch.tensor(180.0))
    if f(dist_diff, torch.pi) < f(dist_diff, torch.pi/2.0):
        x=nn.Parameter(torch.tensor(180.0))
    else:
        x=nn.Parameter(torch.tensor(90.0))
    opt = torch.optim.Adam([x], lr=2)
    losses = []
    pbar = range(500)
    # pbar = tqdm(pbar)
    old_pred = torch.inf
    for i in pbar:
        y = x / 180 * torch.pi
        preds = f(dist_diff, y)
        loss = preds.square()
        loss.backward()
        # print(x.grad)
        opt.step()
        opt.zero_grad()
        # if i % :
        #     for g in opt.param_groups:
        #         g['lr'] = g['lr'] * 0.99
        losses.append(loss.detach())
        # pbar.set_description(str(loss))
        if torch.abs(old_pred - preds) < 0.0001:
            print(i)
            break
        old_pred = preds
    print(x, x % 360), losses

def predict_angle(waveforms, show_tau=False, gcc_phat_method=True):
    window = np.hanning(waveforms.shape[1])

    waveforms = waveforms[[0, 3, 2, 1]]
    tau = torch.zeros(6)
    for i, (idx1, idx2) in enumerate([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]):
        if idx2 - idx1 == 2:
            max_tau = 3.013
        else:
            max_tau = 3.013 / math.sqrt(2.0)
        if gcc_phat_method:
            tau[i], _ = gcc_phat(waveforms[idx2] * window, waveforms[idx1] * window, fs=1, interp=10, max_tau=max_tau)
        else:
            tau[i] = old_calc_time_diffs(waveforms[idx2], waveforms[idx1])
    if show_tau:
        print("tau", tau)
    prediction, _ = try_all_doa(tau)
    return prediction


def old_calc_time_diffs(b, a):
    # time_diffs = torch.zeros(6)
    # for diff_idx, (a_idx, b_idx) in enumerate([(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]):
    #     a = waveforms[a_idx]
    #     b = waveforms[b_idx]
    total_diff = 0.0
    time_diff = torch.zeros(10)
    for sub_idx in range(10):
        a_sub = a[(sub_idx*100):(sub_idx*100)+100]
        b_sub = b[(sub_idx*100):(sub_idx*100)+100]
        # print(a_sub)
        # print(b_sub)
        best = -5
        best_score = torch.inf
        for i in range(-4,5):
            # print("index", i)
            if i < 0:
                # print(a_sub[-i:], b_sub[:i])
                error = (a_sub[-i:] - b_sub[:i]).square().mean()
            elif i > 0:
                # print(a_sub[:-i], b_sub[i:])
                error = (a_sub[:-i] - b_sub[i:]).square().mean()
            else:
                # print(a_sub, b_sub)
                error = (a_sub[:] - b_sub[:]).square().mean()
            # print(error)

            if error < best_score:
                best = i
                best_score = error
        total_diff += best
        time_diff[sub_idx] = best
    # print("mean, std", time_diff.mean(), time_diff.std())
    average_diff = total_diff / 10.0
    # time_diffs[diff_idx] = average_diff
    return average_diff
    # return time_diffs

def mpl_arrow(node_position, angle, length, verbose=False):
    x = [node_position[0], node_position[0] + (length * torch.cos(angle/180.0*torch.pi))]
    y = [node_position[1], node_position[1] + (length * torch.sin(angle/180.0*torch.pi))]
    if verbose:
        print(angle, x, y)
    return x, y