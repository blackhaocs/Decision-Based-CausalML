import numpy as np
import pandas as pd
from IPython.display import display
import torch
import matplotlib.pyplot as plt


def get_uplift_model_aucc(t, yr, yc, roi_pred, quantile=20, title='AUCC'):
    sorted_index = np.argsort(roi_pred)[::-1]

    t = t[sorted_index]
    yr = yr[sorted_index]
    yc = yc[sorted_index]
    roi_pred = roi_pred[sorted_index]

    n_t = np.sum(t)
    n_c = np.sum(~t)
    n = n_t + n_c

    
    nt_list = [0]
    nc_list = [0]

    if n_c > 0: 
        delta_reward = yr[t].mean() - yr[~t].mean()
        delta_cost = yc[t].mean() - yc[~t].mean()
        delta_cost_quantile = delta_cost / quantile
    else:
        n_c = 1
        delta_reward = yr[t].mean()
        delta_cost = yc[t].mean()
        delta_cost_quantile = delta_cost / quantile

    delta_cost_list = [0]
    delta_reward_list = [0]

    t_roi_pred_avg_list = [0]
    c_roi_pred_avg_list = [0]

    cost_t = 0
    reward_t = 0
    cost_c = 0
    reward_c = 0

    i = 0
    j = 1
    while i < n:

        if t[i]:
            cost_t += yc[i]
            reward_t += yr[i]
        else:
            cost_c += yc[i]
            reward_c += yr[i]

        if i >= n-1 or (j < quantile and cost_t / n_t - cost_c / n_c >= delta_cost_quantile * j):
            delta_cost_list.append(cost_t / n_t - cost_c / n_c)
            delta_reward_list.append(reward_t / n_t - reward_c / n_c)
            j += 1

            nt_list.append(np.sum(t[:i + 1]))
            nc_list.append(np.sum(~t[:i + 1]))

            t_roi_pred_avg_list.append(np.mean(roi_pred[:i + 1][t[:i + 1]]))
            c_roi_pred_avg_list.append(np.mean(roi_pred[:i + 1][~t[:i + 1]]))

        i += 1

    
    delta_cost_list = np.array(delta_cost_list)
    delta_reward_list = np.array(delta_reward_list)
    nt_list = np.array(nt_list)
    nc_list = np.array(nc_list)

    aucc = np.sum(delta_reward_list) / (delta_reward * (quantile + 1))

    # plt.plot(delta_cost_list, delta_reward_list, color='r')
    # plt.plot(delta_cost_list, delta_reward / delta_cost * delta_cost_list, color='b')

    # plt.xlabel('delta cost')
    # plt.ylabel('delta reward')
    # plt.title(title)
    # plt.show()

    df_delta_cost = pd.DataFrame(delta_cost_list)
    df_delta_reward = pd.DataFrame(delta_reward_list)
    df_nt = pd.DataFrame(nt_list)
    df_nc = pd.DataFrame(nc_list)
    df_t_roi_pred_avg = pd.DataFrame(t_roi_pred_avg_list)
    df_c_roi_pred_avg = pd.DataFrame(c_roi_pred_avg_list)

    df_delta_cost.rename(columns={0: 'delta_cost'}, inplace=True)
    df_delta_reward.rename(columns={0: 'delta_reward'}, inplace=True)
    df_nt.rename(columns={0: 'n_treatment'}, inplace=True)
    df_nc.rename(columns={0: 'n_control'}, inplace=True)
    df_t_roi_pred_avg.rename(columns={0: 'roi_pred_treatment'}, inplace=True)
    df_c_roi_pred_avg.rename(columns={0: 'roi_pred_control'}, inplace=True)

    df_aucc = pd.concat([df_delta_cost, df_delta_reward, df_nt, df_nc, df_t_roi_pred_avg, df_c_roi_pred_avg], axis=1)
    # display(df_aucc)
    # print("{} = ".format(title), aucc)

    return aucc, delta_cost_list, delta_reward_list, nt_list, nc_list, t_roi_pred_avg_list, c_roi_pred_avg_list, delta_cost


