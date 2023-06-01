# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 13:51:06 2023

@author: mings
"""

import time
from collections import deque
import random
import os

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import gym
import logging
from arguments import get_args
from env import make_vec_envs
from utils.storage import GlobalRolloutStorage, FIFOMemory
from utils.optimization import get_optimizer
from model import RL_Policy, Local_IL_Policy, Neural_SLAM_Module

import algo

import sys
import matplotlib
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

if sys.platform == 'darwin':
    matplotlib.use("tkagg")
import matplotlib.pyplot as plt

# plt.ion()
# fig, ax = plt.subplots(1,4, figsize=(10, 2.5), facecolor="whitesmoke")


args = get_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)


def get_local_map_boundaries(agent_loc, local_sizes, full_sizes):
    loc_r, loc_c = agent_loc
    local_w, local_h = local_sizes
    full_w, full_h = full_sizes

    if args.global_downscaling > 1:
        gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
        gx2, gy2 = gx1 + local_w, gy1 + local_h
        if gx1 < 0:
            gx1, gx2 = 0, local_w
        if gx2 > full_w:
            gx1, gx2 = full_w - local_w, full_w

        if gy1 < 0:
            gy1, gy2 = 0, local_h
        if gy2 > full_h:
            gy1, gy2 = full_h - local_h, full_h
    else:
        gx1.gx2, gy1, gy2 = 0, full_w, 0, full_h

    return [gx1, gx2, gy1, gy2]


def data_scaler(csv):
    """
    Scaling wireless data. Angles to -1 ~ 1. SNR to 0 ~ 1

    Parameters
    ----------
    csv : Pandas DataFrame
        wireless data csv.

    Returns
    -------
    csv : Pandas DataFrame
        Scaled wireless data csv.

    """
    
    angle_columns = ['aoaAzResult_1','aoaAzResult_2','aoaAzResult_3',
            'aoaAzResult_4','aoaAzResult_5','aodAzResult_1',
            'aodAzResult_2','aodAzResult_3','aodAzResult_4',
                     'aodAzResult_5']
    snr_columns = ['snrResult_1','snrResult_2',
           'snrResult_3','snrResult_4','snrResult_5', 'total_snr']
    csv[angle_columns] = csv[angle_columns]/180.0
    snr_max = np.max(csv['total_snr'])
    assert snr_max >= 0.0, "SNR max should > 0"
    csv[snr_columns] = csv[snr_columns]/snr_max
    
    return csv


def load_wireless_data_goal(map_data_dir, map_goal_dir, map_name, tx_num,
                            scale = False):
    """
    Loading wireless data and Target locations (x, y)

    Parameters
    ----------
    map_data_dir : String
        wireless_beam_search.csv dir.
    map_goal_dir : String
        tx_locs.csv dir.
    map_name : StringINFO:root:Start to prepare Episode 

    Returns
    -------
    csv_data : Pandas DataFrame
        wireless data csv.
    list
        Target locs [x, y] in (0, 480) range
    max_snr : float
        Max SNR in dB
    """
    # print((map_data_dir + map_name 
    #                         +"_Tx_"+ tx_num +"_beam_search.csv"))
    csv_data = pd.read_csv(map_data_dir + map_name 
                            +"_Tx_"+ tx_num +"_beam_search.csv")
    snr_columns = ['snrResult_1','snrResult_2',
           'snrResult_3','snrResult_4','snrResult_5']
    
  
    csv_data['total_snr'] = -250.0
    for i in range(len(csv_data)):
        for snr_idx in snr_columns:
            if csv_data[snr_idx].iloc[i] == 0.0:
                # csv_data.at[i, snr_idx] = -np.max(csv_data['snrResult_1'])
                csv_data.at[i, snr_idx] = -250

    for i in range(len(csv_data)):
        if csv_data['linkState'].iloc[i] != 0:
            csv_data.at[i, 'total_snr'] = 10*np.log10(10**(0.1*csv_data[snr_columns[0]].iloc[i]) + 
                                                        10**(0.1*csv_data[snr_columns[1]].iloc[i]) +
                                                        10**(0.1*csv_data[snr_columns[2]].iloc[i]) + 
                                                        10**(0.1*csv_data[snr_columns[3]].iloc[i]) +
                                                        10**(0.1*csv_data[snr_columns[4]].iloc[i]))
    
    # if scale:
    #     csv_data = data_scaler(csv_data)
    max_snr = np.max(csv_data['total_snr'].to_numpy())
    print(max_snr)
    assert max_snr == np.max(csv_data['total_snr'].to_numpy()), "SNR max should = np.max(csv_data['total_snr'].to_numpy())"

    # fix link state:
    for i in range(len(csv_data)):
        if csv_data['linkState'].iloc[i] == 0:
            csv_data.at[i, 'linkState'] = 1
        elif csv_data['linkState'].iloc[i] == 1:
            csv_data.at[i, 'linkState'] = 4
        elif csv_data['linkState'].iloc[i] == 2:
            csv_data.at[i, 'linkState'] = 3
        elif csv_data['linkState'].iloc[i] == 3:
            csv_data.at[i, 'linkState'] = 2
        else:
            print('Error')
            exit()
    
    tx_locs = pd.read_csv(map_goal_dir + map_name +"_tx_pos.csv")
    
    tx_idx_int = int(tx_num) - 1
    x = tx_locs.loc[tx_idx_int, 'x']
    y = tx_locs.loc[tx_idx_int, 'y']
    # first -y
    y = -y
    # for 480 * 480 map
    x = x / 0.05
    y = y / 0.05

    # around to int if needed
    # x = int(x)
    # y = int(y)

    return csv_data, [x, y], max_snr


def read_wireless_measure(map_data, curr_cln, curr_row, 
                    map_resolution, indoor_point_idx, prev_obs = None):
    """
    
    Parameters
    ----------
    map_data : Pandas DataFrame
        wireless data csv.
    curr_cln : int
        agent current column number.
    curr_row : int
        agent currect row number.
    map_resolution : float
        resolution of the map, which is used to convert the coordinates.
    indoor_point_idx : np.array
        indoor points index, (Num_points, 2) row, column
    prev_obs : numpy (16,)
        previous step wireless obs
    Returns
    -------
    wireless_obs : torch tensor with torch.Size([32])
        global obs from wireless (17 old obs, 17 new obs)
    link_state : int value 1, 2, 3, 4
        link state for current location, 4 is LOS, 3 is 1st NLOS, 2 is 2rd NLOS, 1 is outage
    obs : numpy array
        Obs in this step, will use as the last step.

    """
    # the features used as the 'obs'
    wireless_features_columns = ['aoaAzResult_1','aoaAzResult_2','aoaAzResult_3',
        'aoaAzResult_4','aoaAzResult_5','aodAzResult_1',
        'aodAzResult_2','aodAzResult_3','aodAzResult_4',
        'aodAzResult_5','snrResult_1','snrResult_2',
        'snrResult_3','snrResult_4','snrResult_5',
        'total_snr', 'linkState']
    
    r = (curr_row * 100.0/args.map_resolution + 2) / 3
    c = (curr_cln * 100.0/args.map_resolution + 2) / 3
    wireless_obs_df = map_data.loc[(map_data['rxPosInd_1'] == round(r)) & 
                                (map_data['rxPosInd_2'] == round(c))]
    if len(wireless_obs_df['lineIndex'].values) == 0:
        # point not find, round to another points
        diff = np.sum(np.abs(indoor_point_idx - np.array([r, c])), axis = 1)
        idx = np.argmin(diff)
        temp = map_data.iloc[idx]
        link_state = int(temp['linkState'])
        temp = temp[wireless_features_columns].to_numpy() # (17,)
    else:
        temp = wireless_obs_df[wireless_features_columns].to_numpy()[0] # (17,)
        link_state = int(wireless_obs_df['linkState'].values)
    wireless_obs = torch.from_numpy(temp)
    # if prev_obs is None:
        # wireless_obs = torch.from_numpy(np.append(temp, temp))
    # else:
        # wireless_obs = torch.from_numpy(np.append(prev_obs, temp)) # torch.Size([32])

    # if link_state == 1:
    #     link_state = 4
    # elif link_state == 2:
    #     link_state = 3
    # elif link_state == 3:
    #     link_state = 2
    # else:
    #     link_state = 1
    print(f'Log: wireless_obs {wireless_obs}')
    logging.info(f'Log: wireless_obs {wireless_obs}')
    print(f'Log: Current SNR {round(wireless_obs[-1].item(), 2)}, Angle {wireless_obs[0].item()}, LS {link_state}')
    return wireless_obs, link_state, temp


def read_wireless_measure_simple_input(map_data, curr_cln, curr_row, 
                    map_resolution, indoor_point_idx, prev_obs = None):
    """
    Only first path input
    """

    # the features used as the 'obs'
    wireless_features_columns = [
        'aoaAzResult_1','aodAzResult_1','snrResult_1',
        'aoaAzResult_2','aodAzResult_2','snrResult_2',
        # 'aoaAzResult_3','aodAzResult_3','snrResult_3',
        'total_snr', 'linkState'] # 8
    
    r = (curr_row * 100.0/args.map_resolution + 2) / 3
    c = (curr_cln * 100.0/args.map_resolution + 2) / 3
    wireless_obs_df = map_data.loc[(map_data['rxPosInd_1'] == round(r)) & 
                                (map_data['rxPosInd_2'] == round(c))]
    if len(wireless_obs_df['lineIndex'].values) == 0:
        # point not find, round to another points
        diff = np.sum(np.abs(indoor_point_idx - np.array([r, c])), axis = 1)
        idx = np.argmin(diff)
        temp = map_data.iloc[idx]
        link_state = int(temp['linkState'])
        temp = temp[wireless_features_columns].to_numpy() # (5,)
    else:
        temp = wireless_obs_df[wireless_features_columns].to_numpy()[0] # (5,)
        link_state = int(wireless_obs_df['linkState'].values)
    wireless_obs = torch.from_numpy(temp)

    # if link_state == 1:
    #     link_state = 4
    # elif link_state == 2:
    #     link_state = 3
    # elif link_state == 3:
    #     link_state = 2
    # else:
    #     link_state = 1
    print(f'Log: wireless_obs {wireless_obs}')
    logging.info(f'Log: wireless_obs {wireless_obs}')
    print(f'Log: Current SNR {round(wireless_obs[-2].item(), 2)}, Angle {wireless_obs[0].item()}, LS {link_state}')
    return wireless_obs, link_state, temp


# def Reward_function_April5(curr_loc,
#                         map_goal, local_dist,
#                         prev_obs, obs, 
#                         policy_output_ang,
#                         prev_link_state, link_state, max_power, 
#                         flag_done = False):

#     reward = 0
#     # dist of agent to target
#     dist = np.linalg.norm(np.array([curr_loc[0], curr_loc[1]]) 
#                 - np.array([map_goal[0], map_goal[1]]))
#     logging.info(f"Log: Curr distence is: {dist}")

#     angle_diff = abs(prev_obs[0] * 180 - policy_output_ang)

#     minus_power =  (obs[-2] - max_power) # -53 ~ 0
#     if angle_diff > 180:
#         angle_diff = 360 - angle_diff

#     link_punish = 100 * min((link_state - prev_link_state), 0) # -300 ~ 0

#     angle_reward = -angle_diff # -180 ~ 0
#     dist_reward = 600 * np.exp(-0.1*dist)
#     # minus_power_reward = 4 * minus_power

#     if link_state == 4:
#         reward = angle_reward
#     elif link_state == 3:
#         reward = angle_reward * 2 + link_punish
#     else:
#         reward = minus_power * 10 - 180 + link_punish
    
#     logging.info(f"ROBOT CAR's dist: {local_dist}")
#     logging.info(f"Log: power {10 * minus_power - 180}; angle {angle_reward}; link_punish {link_punish}")

#     return reward


# def Reward_function_April8(curr_loc,
#                         map_goal, local_dist,
#                         prev_obs, obs, 
#                         policy_output_ang,
#                         prev_link_state, link_state, max_power, 
#                         flag_done = False):

#     reward = 0
#     # dist of agent to target
#     dist = np.linalg.norm(np.array([curr_loc[0], curr_loc[1]]) 
#                 - np.array([map_goal[0], map_goal[1]]))
#     # logging.info(f"Log: Curr distence is: {dist}")

#     angle_diff = abs(prev_obs[0] - policy_output_ang)

#     minus_power =  (obs[-2] - max_power) # -53 ~ 0
#     if angle_diff > 180:
#         angle_diff = 360 - angle_diff

#     link_punish = 100 * min((link_state - prev_link_state), 0) # -300 ~ 0

#     angle_reward = -angle_diff # -180 ~ 0
#     dist_reward = 600 * np.exp(-0.1*dist)
#     # minus_power_reward = 4 * minus_power

#     if link_state == 4:
#         reward = angle_reward
#     elif link_state == 3:
#         # reward = angle_reward * 2 + link_punish
#         reward = angle_reward * 1.5
#     else:
#         # reward = minus_power * 10 - 180 + link_punish
#         reward = minus_power * 10 - 360

#     logging.info(f"Log: link_state： {link_state}， prev_obs_ang: {prev_obs[0]}, policy_output_ang: {policy_output_ang}, power {10 * minus_power - 360}; angle {angle_reward}; reward： {reward}")

#     return reward

def Reward_function_march18_april9_new(curr_loc,
                        map_goal, local_dist,
                        prev_obs, obs, 
                        policy_output_ang,
                        # policy_output_link_state,
                        prev_link_state, link_state, max_power, 
                        flag_done = False):
    
    
    alpha = 0
    beta = 0
    gamma = 0
    power_imp = 0

    reward = 0

    dist = np.linalg.norm(np.array([curr_loc[0], curr_loc[1]]) 
                - np.array([map_goal[0], map_goal[1]]))
    logging.info(f"Log: Curr distence is: {dist}")

    angle_diff = abs(prev_obs[0] - policy_output_ang)

    # april9_new
    minus_power =  min(obs[-2] - prev_obs[-2], 1) # power_diff
    if angle_diff > 180:
        angle_diff = 360 - angle_diff

    # link_state punishment
    # link_punish = 200 * min((link_state - prev_link_state), 0)
    # March 25
    # link_punish = 100 * (link_state - prev_link_state)
    # March 18 link state reward
    # link_state_diff = - abs((policy_output_link_state * 4) - prev_link_state) * 50 

    angle_reward = 2 * - angle_diff
    dist_reward = 600 * np.exp(-0.1*dist)

    if link_state == 4:
        alpha = 1
        beta = 1
    elif link_state == 3:
        alpha = 2
        beta = 0.5
    else:
        gamma = 100
        power_imp = -720
    
    reward += alpha * angle_reward + beta * dist_reward + gamma * minus_power + power_imp
    
    # logging.info(f"ROBOT CAR's dist: {local_dist}")
    logging.info(f"Log: obs[-2] {obs[-2]}; prev_obs[-2] {prev_obs[-2]}; prev_obs[0] {prev_obs[0]} policy_output_ang {policy_output_ang}")
    logging.info(f"Log: power {gamma * minus_power}; dist {beta * dist_reward}; angle {alpha * angle_reward} rewards {reward}")

    return reward


def Reward_function_march18_april10_new(curr_loc,
                        map_goal, local_dist,
                        prev_obs, obs, 
                        policy_output_ang,
                        # policy_output_link_state,
                        prev_link_state, link_state, max_power, 
                        flag_done = False):
    
    
    alpha = 0
    beta = 0
    gamma = 0
    power_imp = 0

    reward = 0

    dist = np.linalg.norm(np.array([curr_loc[0], curr_loc[1]]) 
                - np.array([map_goal[0], map_goal[1]]))
    logging.info(f"Log: Curr distence is: {dist}")

    angle_diff = abs(prev_obs[0] - policy_output_ang)

    # april9_new
    minus_power =  min(obs[-2] - prev_obs[-2], 1) # power_diff
    if angle_diff > 180:
        angle_diff = 360 - angle_diff

    # link_state punishment
    # link_punish = 200 * min((link_state - prev_link_state), 0)
    # March 25
    # link_punish = 100 * (link_state - prev_link_state)
    # March 18 link state reward
    # link_state_diff = - abs((policy_output_link_state * 4) - prev_link_state) * 50 

    angle_reward = 2 * - angle_diff
    dist_reward = 600 * np.exp(-0.1*dist)

    if link_state == 4:
        alpha = 1
        beta = 1
    # elif link_state == 3:
    #     alpha = 2
    #     beta = 0.5
    else:
        gamma = 100
        power_imp = -360
    
    reward += alpha * angle_reward + beta * dist_reward + gamma * minus_power + power_imp
    
    # logging.info(f"ROBOT CAR's dist: {local_dist}")
    logging.info(f"Log: obs[-2] {obs[-2]}; prev_obs[-2] {prev_obs[-2]}; prev_obs[0] {prev_obs[0]} policy_output_ang {policy_output_ang}")
    logging.info(f"Log: power {gamma * minus_power}; dist {beta * dist_reward}; angle {alpha * angle_reward} rewards {reward}")

    return reward

def get_power_help_ang(map_data, curr_cln, curr_row, 
                    map_resolution, indoor_point_idx, prev_obs = None,
                    offset = 4):
    """
    Get help direction by the power increasing
    Return float angle in [-180, 180]
    """

    snr_column_str = 'total_snr' # 
    
    curr_r = (curr_row * 100.0/args.map_resolution + 2) / 3
    curr_c = (curr_cln * 100.0/args.map_resolution + 2) / 3
    # curr_neg_r = -curr_r
    '''
    get (curr_r, curr_c) total snr in float
    '''
    # read (curr_r, curr_c) data
    wireless_obs_df = map_data.loc[(map_data['rxPosInd_1'] == round(curr_r)) & 
                                (map_data['rxPosInd_2'] == round(curr_c))]
    if len(wireless_obs_df['lineIndex'].values) == 0:
        # not find, get a new (curr_r, curr_c)
        diff = np.sum(np.abs(indoor_point_idx - np.array([curr_r, curr_c])), axis = 1)
        idx = np.argmin(diff)
        temp = map_data.iloc[idx]
        curr_snr = temp[snr_column_str].item()
        # update curr_r and curr_c
        curr_r, curr_c = int(temp['rxPosInd_1']), int(temp['rxPosInd_2'])
    else:
        # find
        curr_snr = wireless_obs_df[snr_column_str].item()
        curr_r, curr_c = round(curr_r), round(curr_c)

    '''
    loop all eight direction (curr_r, curr_c) total snr in float
    '''
    power_ang, around_max = None, None
    # 0 degree, (r, c + 1)
    new_r, new_c = curr_r, curr_c + offset
    new_wireless_obs_df = map_data.loc[(map_data['rxPosInd_1'] == round(new_r)) & 
                                (map_data['rxPosInd_2'] == round(new_c))]
    if not (len(new_wireless_obs_df['lineIndex'].values) == 0):
        # find
        new_snr = new_wireless_obs_df[snr_column_str].item()
        if around_max is None:
            around_max = new_snr
            power_ang = 0
        else:
            if new_snr > around_max:
                around_max = new_snr
                power_ang = 0
    
    # 45 degree, (r - 1, c + 1)
    new_r, new_c = curr_r - offset, curr_c + offset
    new_wireless_obs_df = map_data.loc[(map_data['rxPosInd_1'] == round(new_r)) & 
                                (map_data['rxPosInd_2'] == round(new_c))]
    if not (len(new_wireless_obs_df['lineIndex'].values) == 0):
        # find
        new_snr = new_wireless_obs_df[snr_column_str].item()
        if around_max is None:
            around_max = new_snr
            power_ang = 45
        else:
            if new_snr > around_max:
                around_max = new_snr
                power_ang = 45
    
    # 90 degree, (r - 1, c)
    new_r, new_c = curr_r - offset, curr_c
    new_wireless_obs_df = map_data.loc[(map_data['rxPosInd_1'] == round(new_r)) & 
                                (map_data['rxPosInd_2'] == round(new_c))]
    if not (len(new_wireless_obs_df['lineIndex'].values) == 0):
        # find
        new_snr = new_wireless_obs_df[snr_column_str].item()
        if around_max is None:
            around_max = new_snr
            power_ang = 90
        else:
            if new_snr > around_max:
                around_max = new_snr
                power_ang = 90

    # 135 degree, (r - 1, c - 1)
    new_r, new_c = curr_r - offset, curr_c - offset
    new_wireless_obs_df = map_data.loc[(map_data['rxPosInd_1'] == round(new_r)) & 
                                (map_data['rxPosInd_2'] == round(new_c))]
    if not (len(new_wireless_obs_df['lineIndex'].values) == 0):
        # find
        new_snr = new_wireless_obs_df[snr_column_str].item()
        if around_max is None:
            around_max = new_snr
            power_ang = 135
        else:
            if new_snr > around_max:
                around_max = new_snr
                power_ang = 135

    # 180 / -180 degree, (r, c - 1)
    new_r, new_c = curr_r, curr_c - offset
    new_wireless_obs_df = map_data.loc[(map_data['rxPosInd_1'] == round(new_r)) & 
                                (map_data['rxPosInd_2'] == round(new_c))]
    if not (len(new_wireless_obs_df['lineIndex'].values) == 0):
        # find
        new_snr = new_wireless_obs_df[snr_column_str].item()
        if around_max is None:
            around_max = new_snr
            if random.uniform(-1, 1) >= 0:
                power_ang = 180
            else:
                power_ang = -180
        else:
            if new_snr > around_max:
                around_max = new_snr
                if random.uniform(-1, 1) >= 0:
                    power_ang = 180
                else:
                    power_ang = -180
    
    # -135 degree, (r + 1, c - 1)
    new_r, new_c = curr_r + offset, curr_c - offset
    new_wireless_obs_df = map_data.loc[(map_data['rxPosInd_1'] == round(new_r)) & 
                                (map_data['rxPosInd_2'] == round(new_c))]
    if not (len(new_wireless_obs_df['lineIndex'].values) == 0):
        # find
        new_snr = new_wireless_obs_df[snr_column_str].item()
        if around_max is None:
            around_max = new_snr
            power_ang = -135
        else:
            if new_snr > around_max:
                around_max = new_snr
                power_ang = -135

    # -90 degree, (r + 1, c)
    new_r, new_c = curr_r + offset, curr_c
    new_wireless_obs_df = map_data.loc[(map_data['rxPosInd_1'] == round(new_r)) & 
                                (map_data['rxPosInd_2'] == round(new_c))]
    if not (len(new_wireless_obs_df['lineIndex'].values) == 0):
        # find
        new_snr = new_wireless_obs_df[snr_column_str].item()
        if around_max is None:
            around_max = new_snr
            power_ang = -90
        else:
            if new_snr > around_max:
                around_max = new_snr
                power_ang = -90

    # -45 degree, (r + 1, c + 1)
    new_r, new_c = curr_r + offset, curr_c + offset
    new_wireless_obs_df = map_data.loc[(map_data['rxPosInd_1'] == round(new_r)) & 
                                (map_data['rxPosInd_2'] == round(new_c))]
    if not (len(new_wireless_obs_df['lineIndex'].values) == 0):
        # find
        new_snr = new_wireless_obs_df[snr_column_str].item()
        if around_max is None:
            around_max = new_snr
            power_ang = -45
        else:
            if new_snr > around_max:
                around_max = new_snr
                power_ang = -45        

    logging.info(f'Log: Help Power_ang {power_ang}, power_ang_snr {around_max}, current_loc_snr {curr_snr}')
    print(f'Log: Help Power_ang {power_ang}, power_ang_snr {around_max}, current_loc_snr {curr_snr}')
    assert around_max is not None, "around_max should not be None"


    return power_ang





# def Reward_function_march18_Ming(curr_loc,
#                         map_goal, local_dist,
#                         prev_obs, obs, 
#                         policy_output_ang,
#                         # policy_output_link_state,
#                         prev_link_state, link_state, max_power, 
#                         flag_done = False):
    
    
#     alpha = 0
#     beta = 0
#     gamma = 0
#     power_imp = -360

#     reward = 0

#     dist = np.linalg.norm(np.array([curr_loc[0], curr_loc[1]]) 
#                 - np.array([map_goal[0], map_goal[1]]))
#     logging.info(f"Log: Curr distence is: {dist}")

#     angle_diff = abs(prev_obs[0] - policy_output_ang)

#     minus_power =  obs[-2] - max_power
#     if angle_diff > 180:
#         angle_diff = 360 - angle_diff

#     # link_state punishment
#     link_punish = 200 * min((link_state - prev_link_state), 0)
#     # March 25
#     # link_punish = 100 * (link_state - prev_link_state)
#     # March 18 link state reward
#     # link_state_diff = - abs((policy_output_link_state * 4) - prev_link_state) * 50 

#     angle_reward = 2 * -angle_diff
#     dist_reward = 600 * np.exp(-0.1*dist)
#     # minus_power_reward = 4 * minus_power

#     if link_state == 4:
#         alpha = 1
#         beta = 1
#     elif link_state == 3:
#         alpha = 2
#         beta = 0.5
#     else:
#         gamma = 16
    
#     reward += alpha * angle_reward + beta * dist_reward + gamma * minus_power + link_punish
    
#     # logging.info(f"ROBOT CAR's dist: {local_dist}")
#     logging.info(f"Log: power {gamma * minus_power}; dist {beta * dist_reward}; angle {alpha * angle_reward}; link_punish {link_punish}, alpha {alpha}, beta {beta}, gamma {gamma} rewards")

#     return reward





def main():
    # Setup Logging
    log_dir = "/data2/y2lchong/{}/models/{}/".format(args.dump_location, args.exp_name)
    dump_dir = "/data2/y2lchong/{}/dump/{}/".format(args.dump_location, args.exp_name)
    summary_file_path = "/data2/y2lchong/{}/log/{}/".format(args.dump_location, args.exp_name)
    writer = SummaryWriter(summary_file_path)
    map_data_dir = "{}data/".format(args.map_location)
    map_goal_dir = "{}goal/".format(args.map_location)
    
    # map_name = 'Bowlus'
    # map_name = 'Adrian'
    map_name = 'Woonsocket'
    tx_num = '6'
    map_data_df, map_goal, max_snr = load_wireless_data_goal(map_data_dir, map_goal_dir,
                                                 map_name, tx_num)
    indoor_point_idx = np.array(map_data_df[['rxPosInd_1', 'rxPosInd_2']].values)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists("{}/images/".format(dump_dir)):
        os.makedirs("{}/images/".format(dump_dir))

    logging.basicConfig(
        filename=log_dir + 'train.log',
        level=logging.INFO)
    print("Dumping at {}".format(log_dir))
    print(args)
    logging.info(args)

    # Logging and loss variables
    num_scenes = args.num_processes
    print(f'LOG: Num of scense =  {num_scenes}') # this is 1
    # assert num_scenes == 1, "num_scenes should be 1 or need to correct the g reward"
    num_episodes = int(args.num_episodes)
    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")
    policy_loss = 0

    best_cost = 100000
    costs = deque(maxlen=1000)
    exp_costs = deque(maxlen=1000)
    pose_costs = deque(maxlen=1000)

    g_masks = torch.ones(num_scenes).float().to(device)
    l_masks = torch.zeros(num_scenes).float().to(device)

    best_local_loss = np.inf
    best_g_reward = -np.inf

    if args.eval:
        traj_lengths = args.max_episode_length // args.num_local_steps
        explored_area_log = np.zeros((num_scenes, num_episodes, traj_lengths))
        explored_ratio_log = np.zeros((num_scenes, num_episodes, traj_lengths))

    g_episode_rewards = deque(maxlen=1000)

    l_action_losses = deque(maxlen=1000)

    g_value_losses = deque(maxlen=1000)
    g_action_losses = deque(maxlen=1000)
    g_dist_entropies = deque(maxlen=1000)

    per_step_g_rewards = deque(maxlen=1000)

    g_process_rewards = np.zeros((num_scenes))

    # Starting environments
    torch.set_num_threads(1)
    envs = make_vec_envs(args)
    obs, infos = envs.reset()

    # Initialize map variables
    ### Full map consists of 4 channels containing the following:
    ### 1. Obstacle Map
    ### 2. Exploread Area
    ### 3. Current Agent Location
    ### 4. Past Agent Locations

    torch.set_grad_enabled(False)

    # Calculating full and local map sizes
    map_size = args.map_size_cm // args.map_resolution
    full_w, full_h = map_size, map_size
    local_w, local_h = int(full_w / args.global_downscaling), \
                       int(full_h / args.global_downscaling)
    # print(f"Local W = {local_w}, Local H = {local_h}")

    # Initializing full and local map
    full_map = torch.zeros(num_scenes, 4, full_w, full_h).float().to(device)
    local_map = torch.zeros(num_scenes, 4, local_w, local_h).float().to(device)

    # Initial full and local pose
    full_pose = torch.zeros(num_scenes, 3).float().to(device)
    local_pose = torch.zeros(num_scenes, 3).float().to(device)

    # Origin of local map
    origins = np.zeros((num_scenes, 3))

    # Local Map Boundaries
    lmb = np.zeros((num_scenes, 4)).astype(int)

    ### Planner pose inputs has 7 dimensions
    ### 1-3 store continuous global agent location
    ### 4-7 store local map boundaries
    planner_pose_inputs = np.zeros((num_scenes, 7))

    def init_map_and_pose():
        full_map.fill_(0.)
        full_pose.fill_(0.)
        full_pose[:, :2] = args.map_size_cm / 100.0 / 2.0

        locs = full_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]

            full_map[e, 2:, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

            lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                              (local_w, local_h),
                                              (full_w, full_h))

            planner_pose_inputs[e, 3:] = lmb[e]
            origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                          lmb[e][0] * args.map_resolution / 100.0, 0.]

        for e in range(num_scenes):
            local_map[e] = full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
            local_pose[e] = full_pose[e] - \
                            torch.from_numpy(origins[e]).to(device).float()

    init_map_and_pose()
    wireless_state_shape = [8]
    # Global policy observation space
    g_observation_space = gym.spaces.Box(0, 1,
                                         (wireless_state_shape), dtype='uint8')

    # # Global policy action space
    # g_action_space = gym.spaces.Box(low=0.0, high=1.0,
    #                                 shape=(2,), dtype=np.float32)
    # April7
    g_action_space = gym.spaces.Box(low=0.0, high=1.0,
                                    shape=(1,), dtype=np.float32)

    # Local policy observation space
    l_observation_space = gym.spaces.Box(0, 255,
                                         (3,
                                          args.frame_width,
                                          args.frame_width), dtype='uint8')

    # Local and Global policy recurrent layer sizes
    l_hidden_size = args.local_hidden_size
    g_hidden_size = args.global_hidden_size

    # slam
    nslam_module = Neural_SLAM_Module(args).to(device)
    slam_optimizer = get_optimizer(nslam_module.parameters(),
                                   args.slam_optimizer)

    # Global policy
    g_policy = RL_Policy(g_observation_space.shape, g_action_space, model_type=1,
                         base_kwargs={'recurrent': args.use_recurrent_global,
                                      'hidden_size': g_hidden_size,
                                      'downscaling': args.global_downscaling,
                                      'using_extras': False
                                      }).to(device)
    g_agent = algo.PPO(g_policy, args.clip_param, args.ppo_epoch,
                       args.num_mini_batch, args.value_loss_coef,
                       args.entropy_coef, lr=args.global_lr, eps=args.eps,
                       max_grad_norm=args.max_grad_norm)
    # print("minibatch",args.num_mini_batch)

    # Local policy
    l_policy = Local_IL_Policy(l_observation_space.shape, envs.action_space.n,
                               recurrent=args.use_recurrent_local,
                               hidden_size=l_hidden_size,
                               deterministic=args.use_deterministic_local).to(device)
    local_optimizer = get_optimizer(l_policy.parameters(),
                                    args.local_optimizer)

    # Storage
    g_rollouts = GlobalRolloutStorage(args.num_global_steps,
                                      num_scenes, g_observation_space.shape,
                                      g_action_space, g_policy.rec_state_size,
                                      1).to(device)

    slam_memory = FIFOMemory(args.slam_memory_size)

    # Loading model
    if args.load_slam != "0":
        print("Loading slam {}".format(args.load_slam))
        state_dict = torch.load(args.load_slam,
                                map_location=lambda storage, loc: storage)
        nslam_module.load_state_dict(state_dict)

    if not args.train_slam:
        nslam_module.eval()

    if args.load_global != "0":
        print("Loading global {}".format(args.load_global))
        state_dict = torch.load(args.load_global,
                                map_location=lambda storage, loc: storage)
        g_policy.load_state_dict(state_dict)

    if not args.train_global:
        g_policy.eval()

    if args.load_local != "0":
        print("Loading local {}".format(args.load_local))
        state_dict = torch.load(args.load_local,
                                map_location=lambda storage, loc: storage)
        l_policy.load_state_dict(state_dict)

    if not args.train_local:
        l_policy.eval()

    # Predict map from frame 1:
    poses = torch.from_numpy(np.asarray(
        [infos[env_idx]['sensor_pose'] for env_idx
         in range(num_scenes)])
    ).float().to(device)

    _, _, local_map[:, 0, :, :], local_map[:, 1, :, :], _, local_pose = \
        nslam_module(obs, obs, poses, local_map[:, 0, :, :],
                     local_map[:, 1, :, :], local_pose)

    # Compute Global policy input
    locs = local_pose.cpu().numpy()
    global_input = torch.zeros(num_scenes, 8, local_w, local_h)
    global_orientation = torch.zeros(num_scenes, 1).long()

    for e in range(num_scenes):
        r, c = locs[e, 1], locs[e, 0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]

        local_map[e, 2:, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.
        global_orientation[e] = int((locs[e, 2] + 180.0) / 5.)

    global_input[:, 0:4, :, :] = local_map.detach()
    global_input[:, 4:, :, :] = nn.MaxPool2d(args.global_downscaling)(full_map)
    
    # TODO: transfer original global_input to global_input_wireless
    # read wireless data for current agent location
    # Ming Feb 28th 2023
    start_cj, start_rj, _, _, _, _, _ = planner_pose_inputs[0]
    global_input_wireless, l_state, _ = read_wireless_measure_simple_input(map_data_df,
                                                  start_cj, start_rj, 
                                                  args.map_resolution, 
                                                  indoor_point_idx,
                                                  None)
    prev_obs = global_input_wireless
    assert global_input_wireless.shape == torch.Size([8]), "Wrong wirelss data"
    #
    
    g_rollouts.obs[0].copy_(global_input_wireless)
    g_rollouts.extras[0].copy_(global_orientation)

    # Run Global Policy (global_goals = Long-Term Goal)
    # April 8
    # g_value, g_action, g_action_log_prob, g_rec_states = \
    g_value, g_action, g_action_log_prob, g_rec_states = \
        g_policy.act(
            g_rollouts.obs[0],
            g_rollouts.rec_states[0],
            g_rollouts.masks[0],
            extras=None,
            deterministic=False,
            true_action = None, # April10
        )
    

    cpu_actions = nn.Sigmoid()(g_action).cpu().numpy()
    """
    Ming
    set the init global goal to the init agent location
    we will see that the blue dot in first frame is same as the agent red triangle
    """
    # global_goals = [[int(action[0]), int(action[1])]
    #                 for action in cpu_actions]
    # April 7
    global_goals = [[action[0]]
                    for action in cpu_actions]
    global_goals2 = [[0, 0]
                    for action in cpu_actions]
    policy_output_ang = global_goals[0][0] * 360 - 180 # (-180 ~ 180) degree
    # policy_output_link_state = global_goals[0][1] # （0~1）

    # TODO: initial previous dist = initial location - map goal
    start_cj, start_rj, _, _, _, _, _ = planner_pose_inputs[0]
    start_j = int(start_cj * 100.0/args.map_resolution), int(start_rj * 100.0/args.map_resolution)

    # previous_l_state = l_state
    # Ming March 2
    # set the init global to the init agent location
    # in which case, the agent will not random move at the first step
    # we will see that the blue dot in first frame is same as the agent red triangle
    
    # converting to local coordinates, first getting the currect origin
    o0, o1, _ = origins[0] # num_scenes = 1
    o0 = o0/0.05
    o1 = o1/0.05

    y_pre = start_j[1] - o1 # first 'G-Goal[0]' is the y, this is not wrong.
    x_pre = start_j[0] - o0
    global_goals2[0][0] = int(y_pre)
    global_goals2[0][1] = int(x_pre)
    print(f"Log: Init Link State {l_state}")
    print(f'Log: Init global_goals {[global_goals2[0][1] + o1, global_goals2[0][0] + o0]}')
  
    # Compute planner inputs
    planner_inputs = [{} for e in range(num_scenes)]
    for e, p_input in enumerate(planner_inputs):
        p_input['goal'] = global_goals2[e]
        p_input['map_pred'] = global_input[e, 0, :, :].detach().cpu().numpy()
        p_input['exp_pred'] = global_input[e, 1, :, :].detach().cpu().numpy()
        p_input['pose_pred'] = planner_pose_inputs[e]

    # Output stores local goals as well as the the ground-truth action
    output = envs.get_short_term_goal(planner_inputs)

    last_obs = obs.detach()
    local_rec_states = torch.zeros(num_scenes, l_hidden_size).to(device)
    start = time.time()

    total_num_steps = -1
    g_reward = 0
    torch.set_grad_enabled(False)

    temp_reward = 0

    g_reward = torch.from_numpy(np.asarray(
        [temp_reward for env_idx
            in range(num_scenes)])).float().to(device)
    print(f"LOG: G Reward = {g_reward}")
    logging.info(f"LOG: G Reward = {g_reward}")

    # Lag
    policy_output_ang_prev = policy_output_ang
    # policy_output_link_state_prev = policy_output_link_state
    x_gc_prev = global_goals2[0][1] + o1
    y_gc_prev = global_goals2[0][0] + o0
    logging.info(f"LOG: Initial prev angle and distance = ang: {policy_output_ang_prev}, x: {x_gc_prev}, y: {y_gc_prev}")

    reward_episode_sum = []
    reward_episode_best = -np.inf
    avg_reward = 0
    
    window = deque([], maxlen=args.window_size)


    logging.info(f"===========================================================================================")
    print(f"Before we go into the TWO FOR LOOP {planner_pose_inputs[0]}")
    logging.info(f"Before we go into the TWO FOR LOOP {planner_pose_inputs[0]}")
    logging.info(f"===========================================================================================")

    for ep_num in range(num_episodes):
        
        print(f"Episode {ep_num + 1}")
        logging.info(f"Episode {ep_num + 1}")
        print(f"Loc 1 {planner_pose_inputs[0]}")
        logging.info(f"Loc 1 {planner_pose_inputs[0]}")
        reward_episode = 0
        flag_done = False


        for step in range(args.max_episode_length):
            total_num_steps += 1

            g_step = (step // args.num_local_steps) % args.num_global_steps
            eval_g_step = step // args.num_local_steps + 1
            l_step = step % args.num_local_steps

            # ------------------------------------------------------------------
            # Local Policy
            del last_obs
            last_obs = obs.detach()
            local_masks = l_masks
            local_goals = output[:, :-1].to(device).long()

            if args.train_local:
                torch.set_grad_enabled(True)

            action, action_prob, local_rec_states = l_policy(
                obs,
                local_rec_states,
                local_masks,
                extras=local_goals,
            )

            if args.train_local:
                action_target = output[:, -1].long().to(device)
                policy_loss += nn.CrossEntropyLoss()(action_prob, action_target)
                torch.set_grad_enabled(False)
            l_action = action.cpu()
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Env step
            obs, rew, done, infos = envs.step(l_action)
            done_flag = np.array([flag_done])
            l_masks = torch.FloatTensor([0 if x else 1
                                         for x in done_flag]).to(device)
            g_masks *= l_masks
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Reinitialize variables when episode ends
            if step == args.max_episode_length - 1:  # Last episode step
                init_map_and_pose()
                del last_obs
                last_obs = obs.detach()
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Neural SLAM Module
            if args.train_slam:
                # Add frames to memory
                for env_idx in range(num_scenes):
                    env_obs = obs[env_idx].to("cpu")
                    env_poses = torch.from_numpy(np.asarray(
                        infos[env_idx]['sensor_pose']
                    )).float().to("cpu")
                    env_gt_fp_projs = torch.from_numpy(np.asarray(
                        infos[env_idx]['fp_proj']
                    )).unsqueeze(0).float().to("cpu")
                    env_gt_fp_explored = torch.from_numpy(np.asarray(
                        infos[env_idx]['fp_explored']
                    )).unsqueeze(0).float().to("cpu")
                    env_gt_pose_err = torch.from_numpy(np.asarray(
                        infos[env_idx]['pose_err']
                    )).float().to("cpu")
                    slam_memory.push(
                        (last_obs[env_idx].cpu(), env_obs, env_poses),
                        (env_gt_fp_projs, env_gt_fp_explored, env_gt_pose_err))

            poses = torch.from_numpy(np.asarray(
                [infos[env_idx]['sensor_pose'] for env_idx
                 in range(num_scenes)])
            ).float().to(device)

            _, _, local_map[:, 0, :, :], local_map[:, 1, :, :], _, local_pose = \
                nslam_module(last_obs, obs, poses, local_map[:, 0, :, :],
                             local_map[:, 1, :, :], local_pose, build_maps=True)

            locs = local_pose.cpu().numpy()
            planner_pose_inputs[:, :3] = locs + origins
            local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
            for e in range(num_scenes):
                r, c = locs[e, 1], locs[e, 0]
                loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                int(c * 100.0 / args.map_resolution)]

                local_map[e, 2:, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.
            print(f'LOG: Current Step : {step}')
            logging.info(f'LOG: Current Step : {step}')
            # print(f"Loc 2 {planner_pose_inputs[0]}")
            # logging.info(f"Loc 2 {planner_pose_inputs[0]}")

            # ------------------------------------------------------------------
            # Global Policy
            if l_step == args.num_local_steps - 1:

                # For every global step, update the full and local maps
                for e in range(num_scenes):
                    full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]] = \
                        local_map[e]
                    full_pose[e] = local_pose[e] + \
                                   torch.from_numpy(origins[e]).to(device).float()

                    locs = full_pose[e].cpu().numpy()
                    r, c = locs[1], locs[0]
                    loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                    int(c * 100.0 / args.map_resolution)]

                    lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                                      (local_w, local_h),
                                                      (full_w, full_h))

                    planner_pose_inputs[e, 3:] = lmb[e]
                    origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                                  lmb[e][0] * args.map_resolution / 100.0, 0.]

                    local_map[e] = full_map[e, :,
                                   lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
                    local_pose[e] = full_pose[e] - \
                                    torch.from_numpy(origins[e]).to(device).float()

                locs = local_pose.cpu().numpy()
                for e in range(num_scenes):
                    global_orientation[e] = int((locs[e, 2] + 180.0) / 5.)
                global_input[:, 0:4, :, :] = local_map
                global_input[:, 4:, :, :] = \
                    nn.MaxPool2d(args.global_downscaling)(full_map)
                


                # Add samples to global policy storage
                # TODO: transfer original global_input to global_input_wireless
                                # wireless data processingl_state
                # March 4 Ming
                print(f'LOG: Reading Wireless data step : {step}')
                logging.info(f'LOG: Reading Wireless data step : {step}')
                start_cj, start_rj, _, _, _, _, _ = planner_pose_inputs[0]
                
                logging.info(f'Log: pre read pre_obs: {prev_obs}')
                logging.info(f'LOG: pre read loction: {[start_cj, start_rj]}')
                previous_l_state = l_state
                global_input_wireless, l_state, _ = read_wireless_measure_simple_input(map_data_df,
                                                            start_cj, start_rj, 
                                                            args.map_resolution, 
                                                            indoor_point_idx,
                                                            prev_obs=None)
                
                # April 11
                # Read the power help angle here:
                power_help_ang = get_power_help_ang(map_data_df,
                                                            start_cj, start_rj, 
                                                            args.map_resolution, 
                                                            indoor_point_idx,
                                                            prev_obs=None)

                logging.info(f'Log: after read pre_obs: {prev_obs}')
                # logging.info(f'LOG: after read loction: {[start_cj, start_rj]}')

                print(f"LOG: Link State {l_state}")
                logging.info(f"LOG: Link State {l_state}")

                assert global_input_wireless.shape == torch.Size([8]), "Wrong wirelss data"

                # March 4 Ming
                # Get exploration reward and metrics
                # TODO: Modify Rewards
                """
                global_goals give us gobal coordinates
                using gobal coordinates to compute reward
                then using origin to convert to the local coordinates 
                """
                # computing g_reward by distance
                curr_loc = int(start_cj * 100.0/args.map_resolution), int(start_rj * 100.0/args.map_resolution)
                dist = np.linalg.norm(np.array([curr_loc[0], curr_loc[1]]) 
                                - np.array([map_goal[0], map_goal[1]]))
                print(f"LOG: dist = {dist}")
                logging.info(f"LOG: dist = {dist}")
                # March 12 Ming stop reward ++ 500
                if (dist<10):
                    g_reward += 2000
                    logging.info(f"LOG: DONE Reward Add on {ep_num + 1}!")
                    logging.info(f"g_masks is: {g_masks}")
                if args.eval:
                    g_reward = g_reward*1.0 # Evaluation reward

                g_process_rewards += g_reward.cpu().numpy()
                g_total_rewards = g_process_rewards * \
                                  (1 - g_masks.cpu().numpy())
                g_process_rewards *= g_masks.cpu().numpy()
                per_step_g_rewards.append(np.mean(g_reward.cpu().numpy()))

                if np.sum(g_total_rewards) != 0:
                    for tr in g_total_rewards:
                        g_episode_rewards.append(tr) if tr != 0 else None

                if args.eval:
                    exp_ratio = torch.from_numpy(np.asarray(
                        [infos[env_idx]['exp_ratio'] for env_idx
                         in range(num_scenes)])
                    ).float()

                    for e in range(num_scenes):
                        explored_area_log[e, ep_num, eval_g_step - 1] = \
                            explored_area_log[e, ep_num, eval_g_step - 2] + \
                            g_reward[e].cpu().numpy()
                        explored_ratio_log[e, ep_num, eval_g_step - 1] = \
                            explored_ratio_log[e, ep_num, eval_g_step - 2] + \
                            exp_ratio[e].cpu().numpy()


                g_rollouts.insert(
                    global_input_wireless, g_rec_states,
                    g_action, g_action_log_prob, g_value,
                    g_reward, g_masks, global_orientation
                )
                
                if not flag_done:
                        writer.add_scalar("reward/reward_step", g_reward.item(), ep_num*args.max_episode_length+step+1)
                        reward_episode += g_reward.item()


                    # g_agent = algo.PPO(g_policy, args.clip_param, args.ppo_epoch,
                    #    args.num_mini_batch, args.value_loss_coef,
                    #    args.entropy_coef, lr=args.global_lr, eps=args.eps,
                    #    max_grad_norm=args.max_grad_norm)
                # April 10 help
                if ep_num < 500:
                    if global_input_wireless[-1].item() == 3 or global_input_wireless[-1].item() == 4:
                    # if global_input_wireless[-1].item() == 4:
                        print("help with obs[0]")
                        g_agent.lr = args.global_lr * 5.0
                        # logging.info(f"LOG: g_action, {g_action}, {type(g_action)}")

                        # g_action = np.log(a/(1-a))
                        # logging.info(f"LOG: help, {global_input_wireless[-1].item()}, {global_input_wireless[0].item()}")
                        # a = (global_input_wireless[0].item() + 180) / 360.0 + random.uniform(-0.05, 0.05)
                        a = (global_input_wireless[0].item() + 180) / 360.0
                        a = max(a, 10**(-8))
                        a = min(a, 1-10**(-8))
                        # logging.info(f"LOG: g_action_Sig, {a}, reverse_a, {np.log(a/(1-a))}")
                        g_action_true = np.log(a/(1-a))
                        # g_action.fill_(np.log(a/(1-a)))

                        # logging.info(f"LOG: g_action_after_fill, {g_action}, {type(g_action)}, back {nn.Sigmoid()(g_action).cpu().numpy()}")

                        # global_goals = [[(global_input_wireless[0].item() + 180) / 360.0 + random.uniform(-0.1, 0.1)]
                                    # for action in cpu_actions]
                        # global_goals = [[a]
                                    # for action in cpu_actions]
                        # logging.info(f"LOG: global_goals, {(global_input_wireless[0].item() + 180) / 360.0}")
                        # step_d = 8

                        # fill g_action_log_prob
                        # logging.info(f"LOG: g_action_log_prob, {g_action_log_prob}, {type(g_action_log_prob)}")
                        # g_action_log_prob = g_dist.log_probs(g_action)
                        # logging.info(f"LOG: g_action_log_prob new, {g_action_log_prob}, {type(g_action_log_prob)}")

                    else: # link state = 1, 2, 3
                        print("help with power angle")
                        logging.info(f"help with power angle")
                        g_agent.lr = args.global_lr * 5.0
                        a = (power_help_ang + 180) / 360.0
                        a = max(a, 10**(-8))
                        a = min(a, 1-10**(-8))
                        g_action_true = np.log(a/(1-a))
                    
                else:
                    g_action_true = None
                    g_agent.lr = args.global_lr

                # Sample long-term goal from global policy
                # April8
                # g_value, g_action, g_action_log_prob, g_rec_states = \
                g_value, g_action, g_action_log_prob, g_rec_states = \
                    g_policy.act(
                        g_rollouts.obs[g_step + 1],
                        g_rollouts.rec_states[g_step + 1],
                        g_rollouts.masks[g_step + 1],
                        # true_action,
                        extras=None,
                        deterministic=False,
                        true_action = g_action_true # April 10
                    )
                
                cpu_actions = nn.Sigmoid()(g_action).cpu().numpy()
                writer.add_scalars('actions',{'angle': cpu_actions[0][0],
                                              'distance': 0},
                                              ep_num*args.max_episode_length+step+1)

                
                # ------------------------------------------------------------------
                """
                Ming
                Ming March 2
                new converting of global_goals
                global_goals[0] is the direction angle of the next step
                global_goals[1] is the next step distance
                """
                # global_goals = [[action[0],
                #                  action[1]]
                #                 for action in cpu_actions]
                # April7
                global_goals = [[action[0]]
                                for action in cpu_actions]
                global_goals2 = [[0, 0]
                                for action in cpu_actions]
                
                if global_input_wireless[-1].item() == 3 or global_input_wireless[-1].item() == 4:
                    step_d = 10
                else:
                    step_d = 25 # 15

                global_goals[0][0] = global_goals[0][0] * 360 - 180 # (-180 ~ 180) degree
                policy_output_ang = global_goals[0][0]
                # policy_output_link_state = global_goals[0][1]
                # March 18 Ming
                # if global_goals[0][1] >= 0.5:
                #     global_goals[0][1] = 0.2 * args.walking_distance 
                # else:
                #     global_goals[0][1] = 0.3 * args.walking_distance # walking distance
                # print(f'Log: Global_Goal Angle {round(global_goals[0][0], 2)}, walk_d {round(global_goals[0][1], 2)}')
                print(f'Log: Global_Goal Angle {round(global_goals[0][0], 2)}')
                # logging.info(f'Log: Global_Goal Angle {round(global_goals[0][0], 2)}, walk_d {round(global_goals[0][1], 2)}')
                logging.info(f'Log: Global_Goal Angle {round(global_goals[0][0], 2)}')
                # converting to local coordinates, first getting the currect origin
                start_j = int(start_cj * 100.0/args.map_resolution), int(start_rj * 100.0/args.map_resolution)
                o0, o1, _ = origins[0]
                o0 = o0/0.05
                o1 = o1/0.05

                """
                Verify Ming
                """
                if args.verify:
                    aoa2 = global_input_wireless[0].item() * 180
                else:# Training MIng
                    aoa2 = global_goals[0][0]
                    # step_d = global_goals[0][1]
                    # step_d = 20 # April 5
                
                if aoa2>=0 and aoa2<=90:
                    x_lc = np.cos(np.deg2rad(aoa2))
                    y_lc = -np.sin(np.deg2rad(aoa2))
        
                if aoa2>90 and aoa2<=180:
                    aoa2 = 180-aoa2
                    x_lc = -np.cos(np.deg2rad(aoa2))
                    y_lc = -np.sin(np.deg2rad(aoa2))

                if aoa2<0 and aoa2>=-90:
                    aoa2 = np.absolute(aoa2)
                    x_lc = np.cos(np.deg2rad(aoa2))
                    y_lc = np.sin(np.deg2rad(aoa2))

                if aoa2<-90 and aoa2>=-180: 
                    aoa2 = 180-np.absolute(aoa2)
                    x_lc = -np.cos(np.deg2rad(aoa2))
                    y_lc = np.sin(np.deg2rad(aoa2))

                x_gc = start_j[0] + step_d*x_lc
                y_gc = start_j[1] + step_d*y_lc
                global_goals2[0] = [int(((y_gc) - o1)), int((x_gc) - o0)]

                # print(f'Log: Global transformed {np.array([x_gc, y_gc])}')
                # logging.info(f'Log: Global transformed {np.array([x_gc, y_gc])}')
                # print(f'Log: Truth {np.array([map_goal[0], map_goal[1]])}')
                # logging.info(f'Log: Truth {np.array([map_goal[0], map_goal[1]])}')

                #------------------------------------------------------------------------March12
                if not flag_done:
                    # start_cj, start_rj, _, _, _, _, _ = planner_pose_dummy_inputs[0]
                    start_cj, start_rj, _, _, _, _, _ = planner_pose_inputs[0]
                    
                    curr_loc = int(start_cj * 100.0/args.map_resolution), int(start_rj * 100.0/args.map_resolution)
                    dist = np.linalg.norm(np.array([curr_loc[0], curr_loc[1]]) 
                                    - np.array([map_goal[0], map_goal[1]]))
                    
                    if (dist<10):
                        flag_done = True
                        logging.info(f"LOG: DONE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print(f"LOG: early dist = {dist}")
                    logging.info(f"LOG: early dist = {dist}, flag = {flag_done}")
                #------------------------------------------------------------------------March12

                # temp_reward = Reward_function_April8(
                temp_reward = Reward_function_march18_april9_new(
                    [x_gc_prev, y_gc_prev],
                    map_goal, dist, prev_obs, global_input_wireless,
                    policy_output_ang_prev, 
                    # policy_output_link_state_prev,
                    previous_l_state, l_state, max_snr, flag_done = flag_done)

                
                # Lag
                logging.info(f"|||------------------------------------------------------------------------------------------------------------|||")
                logging.info(f"LOG: Prev angle and distance = ang: {policy_output_ang_prev}, x: {x_gc_prev}, y: {y_gc_prev}, true_angle: {prev_obs[0]}")
                logging.info(f"LOG: Current angle and distance = ang: {policy_output_ang}, x: {x_gc}, y: {y_gc}, true_angle: {global_input_wireless[0]}")
                logging.info(f"LOG: Others infos: previous_l_state: {previous_l_state}, l_state: {l_state}, curr_snr (max=1): {global_input_wireless[-1]}")
                
                policy_output_ang_prev = policy_output_ang
                x_gc_prev = x_gc
                y_gc_prev = y_gc
                prev_obs = global_input_wireless
                # March 18 Ming
                # policy_output_link_state_prev = policy_output_link_state

                # """
                # Verify Ming
                # """
                if flag_done:
                    temp_reward = 2000

                g_reward = torch.from_numpy(np.asarray(
                    [temp_reward for env_idx
                        in range(num_scenes)])).float().to(device)
                print(f"LOG: G Reward = {g_reward}")
                logging.info(f"LOG: G Reward = {g_reward}")

                logging.info(f"|||------------------------------------------------------------------------------------------------------------|||")

                # g_reward = 0
                g_masks = torch.ones(num_scenes).float().to(device)

            # Get short term goal
            planner_inputs = [{} for e in range(num_scenes)]
            for e, p_input in enumerate(planner_inputs):
                p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
                p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
                p_input['pose_pred'] = planner_pose_inputs[e]
                p_input['goal'] = global_goals2[e]

            output = envs.get_short_term_goal(planner_inputs)
            # ------------------------------------------------------------------

            ### TRAINING
            torch.set_grad_enabled(True)
            # ------------------------------------------------------------------
            # Train Neural SLAM Module
            if args.train_slam and len(slam_memory) > args.slam_batch_size:
                for _ in range(args.slam_iterations):
                    inputs, outputs = slam_memory.sample(args.slam_batch_size)
                    b_obs_last, b_obs, b_poses = inputs
                    gt_fp_projs, gt_fp_explored, gt_pose_err = outputs

                    b_obs = b_obs.to(device)
                    b_obs_last = b_obs_last.to(device)
                    b_poses = b_poses.to(device)

                    gt_fp_projs = gt_fp_projs.to(device)
                    gt_fp_explored = gt_fp_explored.to(device)
                    gt_pose_err = gt_pose_err.to(device)

                    b_proj_pred, b_fp_exp_pred, _, _, b_pose_err_pred, _ = \
                        nslam_module(b_obs_last, b_obs, b_poses,
                                     None, None, None,
                                     build_maps=False)
                    loss = 0
                    if args.proj_loss_coeff > 0:
                        proj_loss = F.binary_cross_entropy(b_proj_pred,
                                                           gt_fp_projs)
                        costs.append(proj_loss.item())
                        loss += args.proj_loss_coeff * proj_loss

                    if args.exp_loss_coeff > 0:
                        exp_loss = F.binary_cross_entropy(b_fp_exp_pred,
                                                          gt_fp_explored)
                        exp_costs.append(exp_loss.item())
                        loss += args.exp_loss_coeff * exp_loss

                    if args.pose_loss_coeff > 0:
                        pose_loss = torch.nn.MSELoss()(b_pose_err_pred,
                                                       gt_pose_err)
                        pose_costs.append(args.pose_loss_coeff *
                                          pose_loss.item())
                        loss += args.pose_loss_coeff * pose_loss

                    if args.train_slam:
                        slam_optimizer.zero_grad()
                        loss.backward()
                        slam_optimizer.step()

                    del b_obs_last, b_obs, b_poses
                    del gt_fp_projs, gt_fp_explored, gt_pose_err
                    del b_proj_pred, b_fp_exp_pred, b_pose_err_pred

            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Train Local Policy
            if (l_step + 1) % args.local_policy_update_freq == 0 \
                    and args.train_local:
                local_optimizer.zero_grad()
                policy_loss.backward()
                local_optimizer.step()
                l_action_losses.append(policy_loss.item())
                policy_loss = 0
                local_rec_states = local_rec_states.detach_()
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Train Global Policy
            # if g_step % args.num_global_steps == args.num_global_steps - 1 \
            # March5 Ming
            if g_step % args.num_global_steps == args.num_global_steps - 1 \
                    and l_step == args.num_local_steps - 1:
            # if l_step == args.num_local_steps - 1 and step != args.max_episode_length - 1:
                logging.info(f"Train on {step}")
            # if l_step == args.num_local_steps - 1:
                if args.train_global:
                    print("Training Local in global step: ",g_step," local step: ",l_step)
                    g_next_value = g_policy.get_value(
                        g_rollouts.obs[-1],
                        g_rollouts.rec_states[-1],
                        g_rollouts.masks[-1],
                        extras=None
                    ).detach()

                    g_rollouts.compute_returns(g_next_value, args.use_gae,
                                               args.gamma, args.tau)
                    g_value_loss, g_action_loss, g_dist_entropy = \
                        g_agent.update(g_rollouts)
                    g_value_losses.append(g_value_loss)
                    g_action_losses.append(g_action_loss)
                    g_dist_entropies.append(g_dist_entropy)
                g_rollouts.after_update()
            # ------------------------------------------------------------------

            # Finish Training
            torch.set_grad_enabled(False)
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Logging
            if (total_num_steps % args.log_interval == 0) and (not flag_done):
                end = time.time()
                time_elapsed = time.gmtime(end - start)
                log = " ".join([
                    "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
                    "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
                    "num timesteps {},".format(total_num_steps *
                                               num_scenes),
                    "FPS {},".format(int(total_num_steps * num_scenes \
                                         / (end - start)))
                ])

                log += "\n\tRewards:"

                if len(g_episode_rewards) > 0:
                    log += " ".join([
                        " Global step mean/med rew:",
                        "{:.4f}/{:.4f},".format(
                            np.mean(per_step_g_rewards),
                            np.median(per_step_g_rewards)),
                        " Global eps mean/med/min/max eps rew:",
                        "{:.3f}/{:.3f}/{:.3f}/{:.3f},".format(
                            np.mean(g_episode_rewards),
                            np.median(g_episode_rewards),
                            np.min(g_episode_rewards),
                            np.max(g_episode_rewards))
                    ])

                log += "\n\tLosses:"

                if args.train_local and len(l_action_losses) > 0:
                    log += " ".join([
                        " Local Loss:",
                        "{:.3f},".format(
                            np.mean(l_action_losses))
                    ])

                if args.train_global:
                    log += " ".join([
                        " Global Loss value/action/dist:",
                        "{:.3f}/{:.3f}/{:.3f},".format(
                            np.mean(g_value_losses),
                            np.mean(g_action_losses),
                            np.mean(g_dist_entropies))
                    ])
                    writer.add_scalar("g_action_losses", np.mean(g_action_losses), ep_num*args.max_episode_length+step+1)

                if args.train_slam and len(costs) > 0:
                    log += " ".join([
                        " SLAM Loss proj/exp/pose:"
                        "{:.4f}/{:.4f}/{:.4f}".format(
                            np.mean(costs),
                            np.mean(exp_costs),
                            np.mean(pose_costs))
                    ])

                print(log)
                logging.info(log)
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            if not flag_done:
                # Save best models
                if (total_num_steps * num_scenes) % args.save_interval < \
                        num_scenes:

                    # Save Global Policy Model
                    torch.save(g_policy.state_dict(),
                                os.path.join(log_dir, "model_temp.global"))

                # Save periodic models
                if (total_num_steps * num_scenes) % args.save_periodic < \
                        num_scenes:
                    step = total_num_steps * num_scenes
                    if args.train_slam:
                        torch.save(nslam_module.state_dict(),
                                os.path.join(dump_dir,
                                                "periodic_{}.slam".format(step)))
                    if args.train_local:
                        torch.save(l_policy.state_dict(),
                                os.path.join(dump_dir,
                                                "periodic_{}.local".format(step)))
                    if args.train_global:
                        torch.save(g_policy.state_dict(),
                                os.path.join(dump_dir,
                                                "periodic_{}.global".format(step)))
            
            # early stop the training epidode
            # if (l_step == args.num_local_steps - 1) and flag_done:
            #     _, _ = envs.reset()
            #     break
            # ------------------------------------------------------------------
        writer.add_scalar("reward/reward_episode", reward_episode, ep_num + 1)
        window.append(reward_episode)
        avg_reward = np.mean(window)
        if ep_num == 500 + args.window_size:
            reward_episode_best = -np.inf
        # Save best models
        if reward_episode_best < avg_reward:
            torch.save(g_policy.state_dict(),
                        os.path.join(log_dir, "model_best.global." + str(ep_num)))
        
        reward_episode_best = max(reward_episode_best, avg_reward)
        
        reward_episode_sum.append(reward_episode)
        logging.info(f"Episode {ep_num + 1} episode reward {reward_episode}, cumulate mean reward {np.mean(reward_episode_sum)}")
        logging.info(f"Episode {ep_num + 1} best reward {reward_episode_best}, windows {avg_reward}")

        print(f"Episode {ep_num + 1} End")
        logging.info(f"Episode {ep_num + 1} End")      

    # Print and save model performance numbers during evaluation
    if args.eval:
        logfile = open("{}/explored_area.txt".format(dump_dir), "w+")
        for e in range(num_scenes):
            for i in range(explored_area_log[e].shape[0]):
                logfile.write(str(explored_area_log[e, i]) + "\n")
                logfile.flush()

        logfile.close()

        logfile = open("{}/explored_ratio.txt".format(dump_dir), "w+")
        for e in range(num_scenes):
            for i in range(explored_ratio_log[e].shape[0]):
                logfile.write(str(explored_ratio_log[e, i]) + "\n")
                logfile.flush()

        logfile.close()

        log = "Final Exp Area: \n"
        for i in range(explored_area_log.shape[2]):
            log += "{:.5f}, ".format(
                np.mean(explored_area_log[:, :, i]))

        log += "\nFinal Exp Ratio: \n"
        for i in range(explored_ratio_log.shape[2]):
            log += "{:.5f}, ".format(
                np.mean(explored_ratio_log[:, :, i]))

        print(log)
        logging.info(log)


if __name__ == "__main__":
    main()
