import time
import os, yaml
import numpy as np
from time import sleep

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
from scipy.io import loadmat
import scipy.io as sio

from data_process.load_data import *
from data_process.show_data import *

from model.lstm import *
from self_calibration import CGU_cali


YAML_PATH = 'configs/config_CGU_fitness.yaml'
with open(YAML_PATH) as stream:
    try:
        config = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

PATTERN_MODEL_TRAIN = config['PATTERN_MODEL_TRAIN']
MODEL_TRAIN = config['MODEL_TRAIN']
STORE_EXCEL = config['STORE_EXCEL']
DRAW_IMAGE = config['DRAW_IMAGE']
STROE_MODEL = config['STROE_MODEL']

# The dataset folder in VitalSign
DATA_FOLDER_PATH = os.path.abspath(os.path.join(os.getcwd(), config['DATA_FOLDER_PATH'])) 

# Save predict folder
Save_result_folder = config['Save_result_folder']
Save_model_folder = config['Save_model_folder']

# Data account
DATA_ACCOUNT = config['DATA_ACCOUNT']
RADAR_ACCOUNT = config['RADAR_ACCOUNT']
HR_INDEX = config['HR_INDEX'] # 0 read ground truth HR, 1 read ground truth RR
RR_INDEX = config['RR_INDEX'] # 0 read ground truth HR, 1 read ground truth RR

POWER_READ_INDEX = config['POWER_READ_INDEX']

INPUT_SIZE = config['lstm']['input_size']

# model parameters
MODEL_INPUT = config['lstm']['read_feature']
ATTR_INPUT = config['lstm']['attr_size']
MODEL_OUTPUT = config['lstm']['output_size']

LOSS_TYPE = config['loss_func']
EMA_FLAG = config['smooth']

print("feature all size: ", MODEL_INPUT)
print("Lstm input size: ", INPUT_SIZE)
print("Lstm attr size: ", ATTR_INPUT)

print("Loss function: ", LOSS_TYPE)
print("Predict Smooth: ", EMA_FLAG)

EXCEL_NAME = config['output_excel_name']


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train():

    # model, opts = lstm_model(YAML_PATH)
    # model.to(device)

    # CGU dataset
    name_labels = get_dataset_tester(DATA_FOLDER_PATH)
    print("Total test: ", len(name_labels))
    
    average = [[],[], []] # static, move, all
    move_average = [[], []] #jump, ride
    ignore_list = []
    hr_range = [[], []] # start HR range, max HR
    distances = [[], [], [], []] # Euclidean, correlation coeff, DWT, cosine similarity
    
    # output excel
    output_result = {"Tester":[], 
                     "motion":[],
                     "HR(GT)":[],
                     "HR(Pred)":[],
                    #  "static HR error":[], 
                     "move HR error":[], 
                     "HR error":[],
                     "First HR":[],
                     "Max HR":[],
                     "Euclidean distance":[],
                     "Correlation coefficient":[],
                     "DTW distance":[],
                     "Cosine similarity":[],
                     }
    model_list = {}

    # Cross-validation len(name_labels)
    name_labels = list(name_labels)
    name_labels = name_labels[:]
    name_labels.sort(reverse=True)
    # name_labels.sort(reverse=True)
    print(name_labels)
    Radar_all_test = np.array([0])

    all_data_length = len(name_labels)
    for ni in range(all_data_length):
        # Choose which to train, which to test
        test_name = [name_labels[ni]]
        train_name = []
        for k in range(all_data_length):
            # if i!= k:
            train_name.append(name_labels[k])

        # Prepare Train data
        PowerTrain_ori, VitalTrain_ori, train_path_array, radar_hr, _  = generate_data_CGU_fitness(DATA_FOLDER_PATH, train_name, input_size=MODEL_INPUT-ATTR_INPUT, output_size=MODEL_OUTPUT)        
        PowerTrain = PowerTrain_ori.copy()
        PowerTrain[:, :, :MODEL_INPUT-ATTR_INPUT], _, _ = Power_HR_Normalization(PowerTrain[:, :, :MODEL_INPUT-ATTR_INPUT])
        VitalTrain = VitalTrain_ori

        # Prepare Test data 
        PowerTest_ori, VitalTest_ori, test_path_array, radar_hr, motion_list = generate_data_CGU_fitness(DATA_FOLDER_PATH, test_name, input_size=MODEL_INPUT-ATTR_INPUT, output_size=MODEL_OUTPUT)
        PowerTest = PowerTest_ori.copy()
        PowerTest[:, :, :MODEL_INPUT-ATTR_INPUT], _, _ = Power_HR_Normalization(PowerTest[:, :, :MODEL_INPUT-ATTR_INPUT])
        VitalTest = VitalTest_ori


        Radar_all_test = radar_hr if Radar_all_test.any() == 0 else np.concatenate((Radar_all_test, radar_hr), axis = 0)

        print("\n{0} Read file success!".format(name_labels[ni]))
        print("Power train shape:", PowerTrain.shape)
        print("Heart train shape:", VitalTrain.shape)
        print("Power test shape:", PowerTest.shape) 
        print("Heart test shape:", VitalTest.shape) 
        print("Radar test shape:", Radar_all_test.shape) 
        print("Motion test shape:", np.array(motion_list).shape) 

        if MODEL_TRAIN:
            model, opts = lstm_fitness_model(YAML_PATH)
            # model, opts = lstm_model(YAML_PATH) # The LSTM model

            model.to(device)
            print(model)
            # Model setting
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)   # optimize parameters
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8,weight_decay=1e-4)
            loss_func = vital_loss_func(LOSS_TYPE)
            loss_func1 = vital_loss_func('MAE')

            # Input data setting: [power, hr]
            inputs, targets = torch.from_numpy(PowerTrain).type(torch.FloatTensor),torch.from_numpy(VitalTrain).type(torch.FloatTensor)
            inputs, targets = inputs.to(device), targets.to(device)
            torch_dataset = Data.TensorDataset(inputs,targets)
            loader = Data.DataLoader(
                            dataset=torch_dataset,      # torch TensorDataset format
                            batch_size=opts.batch_size,      # mini batch size
                            shuffle=True,   
                            num_workers=0,                         
                        )
            train_loss, train_count, val_loss, val_count = 0,0,0,0
            train_list, val_list = [], []
            # set val_inputs: power [power, distance], val_targets: hr[hr in linear_interpolation]
            val_inputs, val_targets  = torch.from_numpy(PowerTest).type(torch.FloatTensor),torch.from_numpy(VitalTest).type(torch.FloatTensor)
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
            val_inputs = val_inputs.permute(1, 0, 2)
            val_targets = val_targets.permute(1, 0, 2)

            # begin train
            print("{0} Start to train!..........".format(name_labels[ni]))
            start_time = time.time()
            for epoch in range(0, opts.nums_epoch+1):
                for step, (x, y) in enumerate(loader):
                    
                    model.zero_grad()
                    model.train()
                    
                    x = x.permute(1, 0, 2)
                    y = y.permute(1, 0, 2)

                    prediction, _ = model(x)

                    loss_mae = loss_func1(prediction[:60], y[:60])
                    loss_mse = loss_func(prediction[60:], y[60:])

                    loss = loss_mae+loss_mse

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step() 
                    train_loss += loss.item()
                    train_count+=1

                    model.eval()
                    with torch.no_grad():
                        val_predict, _ = model(val_inputs)
                        loss_ = loss_func(val_predict, val_targets)
                        val_loss += loss_.item()
                        val_count+=1

                train_list.append(train_loss/train_count)
                val_list.append(val_loss/val_count)

                if(epoch% 200 == 0):
                    execution_time = time.time() - start_time
                    print('Epoch %d / %d  time %d   Train loss: %f     Val loss: %f'% (epoch, opts.nums_epoch, execution_time,   train_loss/train_count, val_loss/val_count)) 

            # model.zero_grad()            
            model.eval()

            #--------------------------------------------------------------------------------------------
            print("{0} Start to calculate the error!..........".format(name_labels[ni]))
            
            val_predict, predict_hidden = model(val_inputs)

            val_predict  = val_predict.cpu().detach().numpy()           
            val_inputs  = val_inputs.cpu().detach().numpy()           
            val_targets = val_targets.cpu().detach().numpy()


            val_targets = np.transpose(val_targets, (1, 0, 2))
            val_predict = np.transpose(val_predict, (1, 0, 2))

            print("Test shape: ", val_predict.shape)

            if DRAW_IMAGE:
                draw_hidden_graph(name_labels[ni], predict_hidden, yaml_path=YAML_PATH)
                # draw_loss_graph(name_labels[ni], train_list, val_list, yaml_path=YAML_PATH)
                draw_graph_CGU_fitness(test_path_array, PowerTest_ori, PowerTest, VitalTest_ori, val_predict, yaml_path=YAML_PATH)

            hr_all, hr_static, hr_move = 0, 0, 0
            s_count, m_count = 0, 0
            for pi in range(val_predict.shape[0]):
                path = test_path_array[pi]

                if EMA_FLAG:
                    flag = get_flag(name_labels[ni], motion_list[pi])
                    ema_hr = HR_EMA(val_predict[pi, :, 0], flag, type=1)
                    val_predict[pi, :, 0] = np.array(ema_hr)
                                
                first_hr = val_predict[pi, 0, 0]
                max_hr = np.max(val_predict[pi,:, 0])
                print("Test {0} , \nPredict value: Start HR: {1:.2f}, Max: {2:.2f}".format(pi+1, first_hr, max_hr))
                
                # output excel
                output_result['Tester'].append(name_labels[ni])
                output_result['First HR'].append(first_hr)
                output_result['Max HR'].append(max_hr)

                hr_range[0].append(first_hr)
                hr_range[1].append(max_hr)
                                   
                error_hr = np.abs(np.subtract(val_predict[pi,:, 0], val_targets[pi,:, 0]))
                if int(path) < 40:
                    front_error_hr = error_hr[:180]
                    end_error_hr = error_hr[:]
                    
                    static_error, s_min, s_max = np.mean(front_error_hr), np.min(front_error_hr), np.max(front_error_hr)
                    move_error, m_min, m_max = np.sum(error_hr)/val_predict.shape[1], np.min(end_error_hr), np.max(end_error_hr)

                    move_average[0].append(move_error)

                else:
                    if "jump" in motion_list[pi] or "Jump" in motion_list[pi]:
                        front_error_hr = error_hr[:120]
                        end_error_hr = error_hr[:]
                        static_error, s_min, s_max = np.mean(front_error_hr), np.min(front_error_hr), np.max(front_error_hr)
                        move_error, m_min, m_max = np.sum(error_hr)/val_predict.shape[1], np.min(end_error_hr), np.max(end_error_hr)
                        
                        move_average[0].append(move_error)
                    else:    
                        front_error_hr = error_hr[:60]
                        end_error_hr = error_hr[:]
                        static_error, s_min, s_max = np.mean(front_error_hr), np.min(front_error_hr), np.max(front_error_hr)
                        move_error, m_min, m_max = np.sum(error_hr)/val_predict.shape[1], np.min(end_error_hr), np.max(end_error_hr)

                        move_average[1].append(move_error)
                
                output_result['motion'].append(motion_list[pi])
                error_hr = np.sum(error_hr)/val_predict.shape[1]
                hr_all += error_hr
                
                hr_static += static_error
                s_count += 1
                
                hr_move += move_error
                m_count +=1

                print("HR Error : All: {1:.2f}, move: {2:.2f}\n".format(pi, error_hr, move_error))
                # print("HR Error : All: {1:.2f}, static: {2:.2f}, move: {3:.2f}\n".format(pi, error_hr, static_error, move_error))
                
                # output excel
                output_result['HR(GT)'].append(trans_list2str(val_targets[pi,:, 0]))
                output_result['HR(Pred)'].append(trans_list2str(val_predict[pi,:, 0]))
                # output_result['static HR error'].append("{0:.2f}".format(static_error))
                output_result['move HR error'].append("{0:.2f}".format(move_error))
                output_result['HR error'].append("{0:.2f}".format(move_error))

            model_list[test_name[0]] = model
            if s_count!=0:
                average[0].append(hr_static/s_count)
            if m_count!=0:
                average[1].append(hr_move/m_count)  
            average[2].append(hr_all/val_predict.shape[0])

            print("Output excel...")
            df = pd.DataFrame(output_result)
            writer = pd.ExcelWriter(EXCEL_NAME, engine='xlsxwriter')
            df.to_excel(writer, sheet_name="All",index=False)
            writer.save()


    if MODEL_TRAIN and STORE_EXCEL:
    
        # print("\nStatic: {0:.2f}, {1:.2f}, {2:.2f}".format(np.mean(average[0]), np.min(average[0]), np.max(average[0])))
        print("Move: {0:.2f}, {1:.2f}, {2:.2f}".format(np.sum(average[1])/len(average[1]), np.min(average[1]), np.max(average[1])))
        print("Average: {0:.2f}, {1:.2f}, {2:.2f}".format(np.sum(average[2])/len(average[2]), np.min(average[2]), np.max(average[2])))
        print("Jump error: {0:.2f}, {1:.2f}, {2:.2f}".format(np.sum(move_average[0])/len(move_average[0]), np.min(move_average[0]), np.max(move_average[0])))
        print("Ride error: {0:.2f}, {1:.2f}, {2:.2f}".format(np.sum(move_average[1])/len(move_average[0]), np.min(move_average[1]), np.max(move_average[1])))
        
        print("\nStart HR range: ", int(np.min(hr_range[0])), int(np.max(hr_range[0])))
        print("MAX HR range: ", int(np.min(hr_range[1])), int(np.max(hr_range[1])))


        output_result["Tester"].append("Result")
        output_result["motion"].append("")
        output_result['HR(GT)'].append("")
        output_result['HR(Pred)'].append("")
        # output_result['static HR error'].append("{0:.2f}".format(np.sum(average[0])))
        output_result['move HR error'].append("{0:.2f}".format(np.sum(average[1])/len(average[1])))
        output_result['HR error'].append("{0:.2f}".format(np.sum(average[2])/len(average[2])))
        output_result['First HR'].append("{0:.0f}-{1:.0f}".format(int(np.min(hr_range[0])), int(np.max(hr_range[0]))))
        output_result['Max HR'].append("{0:.0f}-{1:.0f}".format(int(np.min(hr_range[1])), int(np.max(hr_range[1]))))


        df = pd.DataFrame(output_result)
        df1 = pd.DataFrame({
                            # "Static":       ["{0:.2f}".format(np.mean(average[0]))],
                            "Move":	        ["{0:.2f}".format(np.sum(average[1])/len(average[1]))],
                            "Average":	    ["{0:.2f}".format(np.sum(average[2])/len(average[2]))],
                            "Jump error":	["{0:.2f}".format(np.sum(move_average[0])/len(move_average[0]))],
                            "Ride error":	["{0:.2f}".format(np.sum(move_average[1])/len(move_average[1]))],
                            "觀察":          [""],
                            "start HR":     ["{0:.0f}-{1:.0f}".format(int(np.min(hr_range[0])), int(np.max(hr_range[0])))],
                            "Max HR":   	["{0:.0f}-{1:.0f}".format(int(np.min(hr_range[1])), int(np.max(hr_range[1])))],
                            })
        
        writer = pd.ExcelWriter(EXCEL_NAME, engine='xlsxwriter')
        df1.to_excel(writer, sheet_name="Result",index=False)
        df.to_excel(writer, sheet_name="All",index=False)
        writer.save()

    #----------------- Save model parameters -----------------
    result_models_paths = {}
    if MODEL_TRAIN and STROE_MODEL:
        for name in model_list:
            path = "{0}/0622_CGU_{1}.pt".format(Save_model_folder, name)
            result_models_paths[name] = path
            save_model(model_list[name], Save_model_folder, "0622_CGU_{}".format(name))
   
    CGU_cali(EXCEL_NAME)
    # CGU_cali(EXCEL_NAME, Radar_all_test)

    return result_models_paths



if __name__ == '__main__':
    result_models = train()
