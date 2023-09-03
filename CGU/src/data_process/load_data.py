import os, yaml
import random
import glob
import pickle
import numpy as np
import pandas as pd
from scipy.ndimage import interpolation
import matplotlib.pyplot as plt

import torch

def save_model(model, save_folder, file_name):
    if not os.path.exists(save_folder):
        print("make dir")
        os.makedirs(save_folder)

    file_name = save_folder+"/"+file_name+".pt"
    torch.save(model, file_name)


def read_model(Folder_path, model_name):
    model = None

    file_name = Folder_path+"/"+model_name+".pt"
    model = torch.load(file_name)
    model.eval()

    return model


def create_folder(result_folder):
    if not os.path.exists(result_folder):
        print("Make dir: ", result_folder)
        os.makedirs(result_folder)


def data_interpolate(data, total_num):
    tmp_data = data.copy()
    int_tmp_data = np.array([int(i) for i in tmp_data])
    tmp_data = np.delete(tmp_data,np.where(int_tmp_data==-10)) # If exist zero in data, delete
    if tmp_data.shape[0] == 0:
        final_result = np.zeros(total_num)
    else:
        z = total_num/tmp_data.shape[0]
        final_result = interpolation.zoom(tmp_data,z)

    final_result = np.array(final_result, dtype='float')
    return final_result


def remove_noise(data, threshold = 10):
    data = np.delete(data, np.where(data <= threshold)[0], axis=0)

    return data
   

# Generate rawdata read path from folder 
def gene_read_path(radar_account, data_type):
    file_name_list = []
    for i in range(1,radar_account+1):
        name = "{0}{1}{2}".format(data_type[0], str(i).rjust(2,'0'), data_type[1])
        file_name_list.append(name)
    return file_name_list


def Min_Max_Normalization(x):
    max_v = np.max(x)
    min_v = np.min(x)
    # norm_x = (x-min_v)/(max_v-min_v)
    norm_x = (x-np.min(x))/(np.max(x)-np.min(x))
    # return norm_x, np.max(x), np.min(x)
    return norm_x, max_v, min_v


def ZscoreNormalization(x):
    """Z-score normaliaztion"""
    std = np.std(x)
    mean = np.mean(x)

    if int(std * 100000000) != 0:
        x = (x - mean) / std
    else:
        x = (x - mean) / 1

    return x, mean, std


def remove_move_radar_hr(data):
    tmp = data.copy()
    for i in range(tmp.shape[0]):    
        for k in range(tmp.shape[1]):
            power = tmp[i, k, 0]
            if power < 0.5:
                tmp[i, k, 1] = 0


# Z-score normalization
def Power_HR_Normalization(data, power_i=0, hr_i=1):
    tmp = data.copy()
    means = []
    stds = []
    for i in range(tmp.shape[0]):
        means.append([])
        stds.append([])
        for k in range(tmp.shape[2]):
            # if k != tmp.shape[2]-1:
                tmp[i,:,k], mean, std = ZscoreNormalization(tmp[i,:,k])
                means[i].append(mean)
                stds[i].append(std)
    return tmp, np.array(means), np.array(stds)


# Get weight of power per second by status
def get_time_weighted(power, status): # zero for zscore, one for min_max
    move, move_count, release_count = 0, 0, 0
    power = np.array(power)
    weights = np.zeros(power.shape, dtype=float)
    
    power_nor, _, _ = ZscoreNormalization(power)
    params = [0, 3.0, 1.2, 1.15] # static, linear, linear, exponential, for zscore

    for k in range(power.shape[0]): #sequence length
        stat = status[k]

        if stat == 0: # static
            if power_nor[k] > 0:
                power_nor[k] = -1
            weight = params[0]
        elif stat == 1: # static-move
            if power_nor[k] < 1:
                power_nor[k] = 1
            weight = params[1]
        elif stat == 2: # move-move
            if power_nor[k] < 1:
                power_nor[k] = 1
            weight = params[2]
        elif stat == -1: # release
            release_count += 1
            if power_nor[k] > 0:
                power_nor[k] = -1
            if release_count > 5:
                release_count = 4
            weight = pow(params[3], release_count) 

        move += weight*(power_nor[k])

        weights[k] = move

    return weights


# Get target state per second
# flag:
#   0 for 3 min static + 1 min jump
#   1 for 2 min static + 1 min jump + 1 min static
#   2 for 1 min static + 2 min ride + 1 min static
#
def get_move_status(power, flag=0): 
    length = np.array(power).shape[0]
    status = np.zeros(length)

    if flag == 0: # 3 min static + 1 min jump
        status[:180] = 0    
        status[180:] = 1   
    elif flag == 1: # 2 min static + 1 min jump + 1 min static
        status[:120] = 0    
        status[120:180] = 1  
        status[180:] = -1   
    elif flag == 2: # 1 min static + 2 min ride + 1 min static
        status[:60] = 0    
        status[60:120] = 1    
        status[120:180] = 2   
        status[180:] = -1           
    elif flag ==3: # for baseline
        move_count = 0
        tmp_p, _, _ = ZscoreNormalization(power)
        for pi, p in enumerate(tmp_p):
            if p > 0:
                status[pi] = 1 # move
                move_count +=1
            else:
                status[pi] = 0 # static
                if move_count > 0:
                    move_count -= 1
                    status[pi] = 1 # still move
    return status


# The function for baseline, read HR/Power file content
def readDir(dirPath, pathArray, res, file_name, mode):
    if dirPath[-1] == '/':
        return
    flag = False
    allFiles = []
    if os.path.isdir(dirPath):
        fileList = os.listdir(dirPath)
        fileList.sort()
        for f in fileList:
            if f in file_name:
                flag = True
            f = dirPath+'/'+f
            if os.path.isdir(f):
                subFiles = readDir(f,pathArray,res, file_name, mode)
                allFiles = subFiles + allFiles 
            else:
                if flag:
                    pathArray.append(f)
                    if mode == 'power': # csv
                        csv_data = pd.read_csv(f,engine='python',header=None)
                        data = np.array(csv_data)
                    elif mode == 'vital': #npy
                        data = np.load(f)
                    res.append(data)
                    allFiles.append(f)
                    flag = False
    return allFiles



# ---------- Read baseline dataset ----------
# Input: 
#   Root_folder_path: dataset folder, 
#   data_folders: sub person folder, 
#   file_name" {'file_power_name':[...], 'file_hr_name':[...]},
#   input_size: The input number of model,
#   output_size: The output number of model 
# Output: 
#   model input data, ground truth(ex: hr), relative data paths
#------------------------------------------
def generate_dataset(Root_folder_path, data_folders, file_name, input_size=2, output_size=1):
    inputs = np.array([0]) # [Power, distance, HR, RR]
    targets = np.array([0]) # [HR, RR]
    all_data_path = []
    
    file_power_name = file_name['file_power_name']
    file_hr_name = file_name['file_HR_name']
    
    for name in data_folders:
        # read power and distance from file according ro name_labels as trainset
        pathArray =  []
        power, vital = [], []
        path = "{root}/{name}".format(root=Root_folder_path, name=name)
        readDir(path, pathArray, power, file_power_name, "power")
        power = np.array(power)

        move_status = []
        move_weight = []
        for i in range(power.shape[0]):
            p = power[i, : ,1]
            status = get_move_status(p, flag=3)
            weight = get_time_weighted(p, status)
            # print(status)
            move_status.append(status)
            move_weight.append(weight)
        move_status = np.array(move_status)
        move_weight = np.array(move_weight)

        pathArray =  []
        readDir(path, pathArray, vital, file_hr_name, "vital")
        vital = np.array(vital)
        # vital interpolation
        new_v = np.zeros((vital.shape[0], vital.shape[1], vital.shape[2]))
        # print(vital.shape, new_v.shape)
        new_v[:, :vital.shape[1], :] = vital
        for p in range(new_v.shape[0]):
            new_v[p, vital.shape[1]:, :] = vital[p, -1, :]
       

        tmp_p, tmp_v = power, vital
        # tmp_p, tmp_v = power, vital
        reshape_input, reshape_target = [], []
        for k in range(tmp_p.shape[0]):
            reshape_input.append([])
            reshape_target.append([])
            for n in range(tmp_p.shape[1]):
                if input_size ==1:
                    reshape_input[k].append([tmp_p[k, n, 1]])#[Power]
                elif input_size == 2:
                    reshape_input[k].append(tmp_p[k, n, 1:])#[Power, distance]
                elif input_size == 3:
                    reshape_input[k].append(np.append(tmp_p[k, n, 1:], move_weight[k, n]))#[Power, distance, move_weight]                  
          
                if output_size == 1:
                    reshape_target[k].append([tmp_v[k, n, 0]])#[HR, RR offset 1 second]

                elif output_size == 2:
                    reshape_target[k].append(tmp_v[k, n, :])#[HR, RR offset 1 second]
 
        
        reshape_input = np.array(reshape_input)
        reshape_target = np.array(reshape_target)

        for p in pathArray:
            all_data_path.append(p)
        inputs = reshape_input if inputs.any() == 0 else np.concatenate((inputs, reshape_input), axis = 0)
        targets = reshape_target if targets.any() == 0 else np.concatenate((targets, reshape_target), axis = 0)
    
    return inputs, targets, all_data_path, 0



# ---------- Read CGU dataset ----------

# The function for CGU, get data type
def get_flag(name, motion):
    name = int(name)
    if name < 40 and 'jump' in str(motion).lower():
        return 0
    elif 'jump' in str(motion).lower():
        return 1
    else:
        return 2

# Exponential moving average for ground truth HR,
# To smooth the meandering curve of ground truth.  
def HR_EMA(hr, ema_num, flag=0, type=0):
    if type==0:
        if flag == 0: # for 3 min static, 1 min jump
            pass
        elif flag == 1: # for 2 min static, 1 min jump, 1 min static
            tmp = pd.DataFrame(hr[:60])
            tmp = tmp.ewm(span=20).mean()
            hr[:60] = tmp[0]

            tmp = pd.DataFrame(hr[61:125])
            tmp = tmp.ewm(span=20).mean()
            hr[61:125] = tmp[0]

            tmp = pd.DataFrame(hr[125:])
            tmp = tmp.ewm(span=3).mean()
            hr[125:] = tmp[0]
        elif flag == 2: # for 1 min static, 2 min ride, 1 min static
            tmp = pd.DataFrame(hr[:70])
            tmp = tmp.ewm(span=20).mean()
            hr[:70] = tmp[0]

            tmp = pd.DataFrame(hr[70:])
            tmp = tmp.ewm(span=5).mean()
            hr[70:] = tmp[0]
    else:
        tmp = pd.DataFrame(hr)
        tmp = tmp.ewm(span=ema_num).mean()
        hr = tmp[0]


    return np.array(hr)


def trans_str2list(s):
    data = []
    for k in (s.split(",")):
        if len(k)!=1:
            data.append(float(k))
    return data


def trans_list2str(tmp):
    s = ""
    for i in tmp:
        s+=","+str(i)
    
    return s[1:]


# Read fitness datas of testers
def read_fitness_data(DATA_FOLDER_PATH, name_label):
    csv = DATA_FOLDER_PATH+"/fitness.csv"
    csv_data = pd.read_csv(csv, engine='python')
    # "Tester":[], "Low":[], "Moderate":[], "High":[], "Score":[]
    # print(csv_data)
    
    tester = csv_data["Tester"]
    cv = csv_data["Class"]
    score = csv_data["Score"]
    g_raw = csv_data["Gender"]
    age = csv_data["Age"]
    bmi = csv_data["BMI"]
    gender = []


    cv = np.array(cv, dtype=int) - 2
    score, _, _ = Min_Max_Normalization(score)
    age, _, _ = Min_Max_Normalization(age, min_v=19, max_v=40)
    bmi, _, _ = Min_Max_Normalization(np.array(bmi, dtype=float))


    for i in range(len(g_raw)):
        if "M" in g_raw[i] : #Male
            gender.append(0)
        else: # Female
            gender.append(1)

    data = []
    for ni in range(len(name_label)):
        name = name_label[ni]
        i = list(tester).index(name)
        data.append([cv[i], score[i], age[i], bmi[i], gender[i]])

    data = np.array(data, dtype=float)

    return data


# Read CGU tester
def get_dataset_tester(DATA_FOLDER_PATH):
    csvs = glob.glob(DATA_FOLDER_PATH+"/*-all.csv")
    
    testers = []
    for csv in csvs:
        csv_data = pd.read_csv(csv, engine='python')
        if "move" in csv:
            # move
            testers = set(csv_data['Tester'])

    return list(testers)


#  Read CGU dataset
def readDataset(DATA_FOLDER_PATH):
    
    csvs = glob.glob(DATA_FOLDER_PATH+"/*-all.csv")
    # print(csvs)
    
    testers = []
    datas_list = []
    static_data_list = []
    pathes_list = []
    motion_list = []
    for csv in csvs:
        csv_data = pd.read_csv(csv, engine='python')
        csv_path = csv.replace(DATA_FOLDER_PATH+"/", "")
       
        if "move" in csv:
            # move
            testers = csv_data['Tester']
            for i in range(len(csv_data['GT(HR)'])):
                gt_hr = trans_str2list(csv_data['GT(HR)'][i]) 
                radar_hr = trans_str2list(csv_data['Radar(HR)'][i])
                power = trans_str2list(csv_data['Power'][i])
                dis = trans_str2list(csv_data['Distance'][i])
                datas_list.append([gt_hr, radar_hr, power, dis])
                motion_list.append(csv_data['Motion'][i])

        elif "static" in csv:
            # static
            datas = []
            for i in range(len(csv_data['GT(HR)'])):
                gt_hr = trans_str2list(csv_data['GT(HR)'][i]) 
                radar_hr = trans_str2list(csv_data['Radar(HR)'][i])
                power = trans_str2list(csv_data['Power'][i])
                dis = trans_str2list(csv_data['Distance'][i])
                static_data_list.append([gt_hr, radar_hr, power, dis])  
        
    return datas_list, testers, pathes_list, static_data_list, motion_list


# ---------- Read CGU dataset ----------
# Input: 
#   Root_folder_path: dataset folder, 
#   data_folders: sub person folder, 
#   input_size: The input number of model,
#   output_size: The output number of model 
# Output: 
#   model input data, ground truth(ex: hr), relative data paths
#------------------------------------------
def generate_data_CGU(DATA_FOLDER_PATH, name_label, input_size=3, output_size=1):
    Power_all = np.array([0])
    Vital_all = np.array([0])
    Radar_all = np.array([0])
    all_data_path = []
    motion_list = []
    datas_list, names, _, static_data_list, motion_lists = readDataset(DATA_FOLDER_PATH)
    datas_list = np.array(datas_list)

    # static_data_list = np.array(static_data_list)
    for ni in range(len(names)):
        name = names[ni]
        if name not in name_label:
            continue
        
        all_data_path.append(name)
        motion_list.append(motion_lists[ni])

        motion_datas = np.array(datas_list)
        HR = np.array(motion_datas[ni, 0,:]) # hr ground truth
        tmp = pd.DataFrame(HR)
        tmp = tmp.ewm(span=5).mean()
        HR = np.array(tmp[0])

        length = np.array(motion_datas[ni, 1,:]).shape[0]
        
        radar_HR = np.array(motion_datas[ni, 1,:]) # hr from radar  
        radar_HR = data_interpolate(remove_noise(radar_HR, threshold=60), length)       
        
        power = np.array(motion_datas[ni, 2,:]) # power
        distance = np.array(motion_datas[ni, 3,:]) # power

        power = power - np.min(power)
        move_status = get_move_status(power)
        time_weight = get_time_weighted(power, move_status)

        # print(move_status)
        # print("Time: ", time_weight.shape) # Debug

        new_train = [[]]
        new_target = [[]]
        new_radar = [[]]
        for k in range(power.shape[0]):
            new_radar[0].append([radar_HR[k]])
            
            if input_size == 3:
                new_train[0].append(np.array([power[k], distance[k], time_weight[k]]))
            elif input_size == 2:
                new_train[0].append(np.array([power[k], distance[k]]))
            
            if output_size == 1:
                new_target[0].append([HR[k]])

        new_train = np.array(new_train)
        new_target = np.array(new_target)
        new_radar = np.array(new_radar)

        Power_all = new_train if Power_all.any() == 0 else np.concatenate((Power_all, new_train), axis = 0)
        Vital_all = new_target if Vital_all.any() == 0 else np.concatenate((Vital_all, new_target), axis = 0)
        Radar_all = new_radar if Radar_all.any() == 0 else np.concatenate((Radar_all, new_radar), axis = 0)
    return Power_all, Vital_all, all_data_path, Radar_all, motion_list


# ---------- Read CGU fitness dataset ----------
# Input: 
#   Root_folder_path: dataset folder, 
#   data_folders: sub person folder, 
#   input_size: The input number of model,
#   output_size: The output number of model 
# Output: 
#   model input data(include IPAQ data), ground truth(ex: hr), relative data paths
#------------------------------------------
def generate_data_CGU_fitness(DATA_FOLDER_PATH, name_label, input_size=3, output_size=1):
    Power_all = np.array([0])
    Vital_all = np.array([0])
    Radar_all = np.array([0])
    all_data_path = []
    motion_list = []

    datas_list, names, _, static_data_list, motion_lists = readDataset(DATA_FOLDER_PATH)
    datas_list = np.array(datas_list)
    
    # static_data_list = np.array(static_data_list)
    for ni in range(len(names)):
        name = names[ni]
        if name not in name_label:
            continue

        all_data_path.append(name)
        motion_list.append(motion_lists[ni])
        fitness = read_fitness_data(DATA_FOLDER_PATH, [name])

        motion_datas = np.array(datas_list)
        HR = np.array(motion_datas[ni, 0,:]) # hr ground truth

        flag = get_flag(name, motion_lists[ni])
        HR = HR_EMA(HR, _, flag, type=0) # hr, ema_num, flag=0, type=0

        length = np.array(motion_datas[ni, 1,:]).shape[0]
        
        radar_HR = np.array(motion_datas[ni, 1,:]) # hr from radar  
        radar_HR = data_interpolate(remove_noise(radar_HR, threshold=55), length)       
        
        power = np.array(motion_datas[ni, 2,:]) # power
        distance = np.array(motion_datas[ni, 3,:]) # power

        power = power - np.min(power)

        move_status = get_move_status(power, flag)
        time_weight = get_time_weighted(power, move_status)

        # print("Time: ", time_weight.shape) # Debug

        new_train = [[]]
        new_target = [[]]
        new_radar = [[]]

        for k in range(power.shape[0]):
            new_radar[0].append([radar_HR[k]])

            if radar_HR[k] < 10 :
                radar_HR[k] = 0


            if input_size == 3: # power, distance, move_weight, ipaq score, ipaq class
                tmp = [power[k], distance[k], time_weight[k]]
                # tmp = np.append(tmp, fitness)
                new_train[0].append(np.array(tmp))
            elif input_size == 7: #power, distance, ipaq score, ipaq class, age, BMI, gender
                tmp = [power[k], distance[k]]
                tmp = np.append(tmp, fitness)    
                new_train[0].append(np.array(tmp))
            elif input_size == 8: #power, distance, move_weight, ipaq score, ipaq class, age, BMI, gender
                tmp = [power[k], distance[k], time_weight[k]]
                tmp = np.append(tmp, fitness)    
                new_train[0].append(np.array(tmp))
            else:
                tmp = [power[k], distance[k]]
                new_train[0].append(np.array(tmp))

            if output_size == 1:
                new_target[0].append([HR[k]])


        new_train = np.array(new_train)
        new_target = np.array(new_target)
        new_radar = np.array(new_radar)

        Power_all = new_train if Power_all.any() == 0 else np.concatenate((Power_all, new_train), axis = 0)
        Vital_all = new_target if Vital_all.any() == 0 else np.concatenate((Vital_all, new_target), axis = 0)
        Radar_all = new_radar if Radar_all.any() == 0 else np.concatenate((Radar_all, new_radar), axis = 0)


    return Power_all, Vital_all, all_data_path, Radar_all, motion_list