import os, yaml
import numpy as np

import torch
import torch.utils.data as Data


from data_process.load_data import *
from data_process.show_data import draw_graph
from model.lstm import *
from self_calibration import vital_cali

YAML_PATH = 'configs/baseline_config.yaml'
with open(YAML_PATH) as stream:
    try:
        config = yaml.full_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

MODEL_TRAIN = config['MODEL_TRAIN']
STORE_EXCEL = config['STORE_EXCEL']
DRAW_IMAGE = config['DRAW_IMAGE']
STROE_MODEL = config['STROE_MODEL']

# The dataset folder in VitalSign
DATA_FOLDER_PATH = os.path.abspath(os.path.join(os.getcwd(), config['DATA_FOLDER_PATH'])) 

# Save predict folder
Save_result_folder = config['Save_result_folder']
Save_model_folder = config['Save_model_folder']
EXCEL_NAME = config['output_excel_name']

create_folder("/".join(Save_result_folder.split("/")[:-1]))
create_folder("/".join(Save_model_folder.split("/")[:-1]))
create_folder("/".join(EXCEL_NAME.split("/")[:-1]))

# Data account
DATA_ACCOUNT = config['DATA_ACCOUNT']
RADAR_ACCOUNT = config['RADAR_ACCOUNT']
HR_INDEX = config['HR_INDEX'] # 0 read ground truth HR, 1 read ground truth RR
RR_INDEX = config['RR_INDEX'] # 0 read ground truth HR, 1 read ground truth RR
LOSS_TYPE = config['loss_func']
EMA_NUM = config['smooth']

# model parameters
MODEL_INPUT = config['lstm']['input_size']
MODEL_OUTPUT = config['lstm']['output_size']


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("----------- Config content-----------")
print("MODEL_TRAIN: ", MODEL_TRAIN)
print("STORE_EXCEL: ", STORE_EXCEL)
print("DRAW_IMAGE: ", DRAW_IMAGE)
print("STROE_MODEL: ", STROE_MODEL)
print("Loss function: ", LOSS_TYPE)
print("EMA Smooth: ", EMA_NUM)
print("-------------------------------------\n")



if __name__ == '__main__':
    # Generate read path for each data(person)
    # name_labels = ["p{0}".format(i) for i in range(1, 2)] # folder for each data "p1, p2, ..."
    name_labels = ["p{0}".format(i) for i in range(1, DATA_ACCOUNT+1)] # folder for each data "p1, p2, ..."
    others_labels = ['differentplace/place1', 'differentplace/place2', 'noise']
    for l in others_labels:
        name_labels.append(l)
    file_name = {
        "file_power_name": gene_read_path(RADAR_ACCOUNT, ['radar_', '.csv']),
        "file_HR_name": gene_read_path(RADAR_ACCOUNT, ['HR', '.npy'])}
    model_list = {}

    # for each p1,...p14, diff/place1, diff/place2, noise
    all_p_MSE_hr = {"amb":{"all":[],"fb":[], "lr":[]},
             "per":{"1m":[],"2m":[], "3m":[]},
             "ran":{"1m":[],"2m":[], "3m":[]}} 


    average = [[],[], []] # static, move, all
    hr_range = [[], []] # start HR range, max HR
    
    # output excel
    output_result = {"Tester":[], 
                     "motion":[],
                     "range": [],
                     "HR(GT)":[],
                     "HR(Pred)":[],
                     "static HR error":[], 
                     "move HR error":[], 
                     "HR error":[],
                     "First HR":[],
                     "Max HR":[],
                     }
    
    # Cross-validation len(name_labels)
    all_data_length = len(name_labels)
    for i in range(all_data_length):
        # Choose which to train, which to test
        test_name = [name_labels[i]]
        train_name = []
        for k in range(all_data_length):
            if i!= k:
                train_name.append(name_labels[k])

        # Prepare Train data
        PowerTrain_ori, VitalTrain, train_path_array, _ = generate_dataset(DATA_FOLDER_PATH, train_name, file_name, MODEL_INPUT, MODEL_OUTPUT)
        PowerTrain, _, _ = Power_HR_Normalization(PowerTrain_ori)

        # Prepare Test data
        PowerTest_ori, VitalTest, test_path_array, _ = generate_dataset(DATA_FOLDER_PATH, test_name, file_name, MODEL_INPUT, MODEL_OUTPUT)
        PowerTest, _, _ = Power_HR_Normalization(PowerTest_ori)

        print("\n{0} Read file success! ({1}/{2})".format(name_labels[i], i+1, all_data_length))

        print("Train shape:", PowerTrain.shape)
        print("Vtial train shape:", VitalTrain.shape)
        print("Test shape:", PowerTest.shape)
        print("Vtial test shape:", VitalTest.shape)
        
        if MODEL_TRAIN:
            # Model setting
            model, opts = lstm_model(YAML_PATH)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)   # optimize parameters
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.8, weight_decay=1e-4)
            loss_func = vital_loss_func(LOSS_TYPE)

            print("Vital ada model\n", model)
            
            # Input data setting: [power, hr]
            inputs, targets = torch.from_numpy(PowerTrain).type(torch.FloatTensor),torch.from_numpy(VitalTrain).type(torch.FloatTensor)
            inputs, targets = inputs.to(device), targets.to(device)
            torch_dataset = Data.TensorDataset(inputs,targets)
            loader = Data.DataLoader(
                            dataset=torch_dataset,      # torch TensorDataset format
                            batch_size=opts.batch_size,      # mini batch size
                            shuffle=True,                            
                        )
            train_loss, train_count, val_loss, val_count = 0,0,0,0
            # set val_inputs: power [power, distance], val_targets: hr[hr in linear_interpolation]
            val_inputs, val_targets  = torch.from_numpy(PowerTest).type(torch.FloatTensor),torch.from_numpy(VitalTest).type(torch.FloatTensor)
            val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
            x_ada = torch.zeros(val_inputs.shape)
            y_ada = torch.zeros(val_targets.shape)
            # print(x_ada.shape,y_ada.shape,val_inputs.shape)
            for ii in range(x_ada.shape[0]):
                x_ada[ii,100:150,:] = val_inputs[ii,100:150,:]
                x_ada[ii,150:200,:] = val_inputs[ii,150:200,:]
                # x_ada[ii,100:,:] = val_inputs[ii,100:,:]
                # x_ada[ii,200:,:] = val_inputs[ii,200:,:]
                for jj in range(50):
                    if(opts.adaTrain > 0):
                        # y_ada[ii,250+jj] = val_targets[ii,-100] 
                        # y_ada[ii,200+jj] = val_targets[ii,-50] 
                        y_ada[ii,150+jj] = val_targets[ii,150]
                        y_ada[ii,100+jj] = val_targets[ii,100]

            val_inputs = val_inputs.permute(1, 0, 2)
            val_targets = val_targets.permute(1, 0, 2)

            # begin train
            print("{0} Start to train!..........".format(name_labels[i]))
            for epoch in range(0, opts.nums_epoch+1):
                for step, (x, y) in enumerate(loader):
                    
                    model.zero_grad()
                    model.train()
                    x = x.permute(1, 0, 2)
                    y = y.permute(1, 0, 2)
                    prediction, _ = model(x)
                    loss = loss_func(prediction, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step() 
                    train_loss += loss.item()
                    train_count += 1

                model.eval()
                with torch.no_grad():
                    val_predict, _ = model(val_inputs)
                    loss_ = loss_func(val_predict, val_targets)
                    val_loss += loss_.item()
                    val_count+=1

                if(epoch% 100 == 0):
                    print('Epoch %d / %d     Train loss: %f     Val loss: %f'% (epoch, opts.nums_epoch,  train_loss/train_count, val_loss/val_count)) 

            # Adavtive training
            print("Ada training")
            x_ada,y_ada= x_ada.to(device), y_ada.to(device)
            torch_dataset_ada = Data.TensorDataset(x_ada,y_ada)
            ada_loss = 0.0
            loader1 = Data.DataLoader(
                            dataset=torch_dataset_ada,      # torch TensorDataset format
                            batch_size=opts.batch_size,      # mini batch size
                            shuffle=True,                            
                        )
            for epoch in range(1, opts.nums_ada_epoch+1):
                for step, (x, y) in enumerate(loader1):
                    model.zero_grad()
                    model.train()
                    x = x.permute(1, 0, 2)
                    y = y.permute(1, 0, 2)

                    prediction, _ = model(x)
                    # loss = loss_func(prediction[200:], y[200:])
                    loss = loss_func(prediction[100:200], y[100:200])

                    ada_loss = loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step() 
                    
                print('Epoch %d / %d     ada loss: %f '% (epoch, opts.nums_ada_epoch,  ada_loss)) 
            
            model.zero_grad()
            

            #--------------------------------------------------------------------------------------------
            print("{0} Start to calculate the error!..........".format(name_labels[i]))
            
            val_predict, _ = model(val_inputs)
            val_predict  = val_predict.cpu().detach().numpy()   
            val_inputs  = val_inputs.cpu().detach().numpy()   
            val_targets = val_targets.cpu().detach().numpy()
                    
            # EMA Smooth
            if EMA_NUM!=-1:
                tmp_predict = []
                for ti in range(val_predict.shape[1]):
                    pre = val_predict[:, ti, 0]
                    tmp = pd.DataFrame(pre)
                    tmp = tmp.ewm(span=EMA_NUM).mean()
                    tmp_predict.append(tmp)
                val_predict = np.array(tmp_predict)

            val_targets = np.transpose(val_targets, (1, 0, 2))
            if not (EMA_NUM!=-1): val_predict = np.transpose(val_predict, (1, 0, 2))

            if DRAW_IMAGE:
                draw_graph(test_path_array, Save_result_folder, PowerTest, VitalTest, val_predict, 1, yaml_path=YAML_PATH)


            hr_all, hr_static, hr_move = 0, 0, 0
            s_count, m_count = 0, 0
            for pi in range(val_predict.shape[0]):
                path = test_path_array[pi]
                first_hr = val_predict[pi, 0, 0]
                max_hr = np.max(val_predict[pi,:, 0])
                print("Test {0} , \nPredict value: Start HR: {1:.2f}, Max: {2:.2f}".format(pi+1, first_hr, max_hr))

                hr_range[0].append(first_hr)
                hr_range[1].append(max_hr)
                                   
                error_hr = np.abs(np.subtract(val_predict[pi,:, 0], val_targets[pi,:, 0]))
                front_error_hr = error_hr[:110]
                end_error_hr = error_hr[110:]
                
                static_error = np.mean(front_error_hr)
                move_error = np.mean(end_error_hr)
                error_hr = np.mean(error_hr)

                hr_all += error_hr
                
                hr_static += static_error
                s_count += 1
                
                hr_move += move_error
                m_count +=1
                
                print("HR Error : {:.2f}".format(error_hr))

                model_list[test_name[0]] = model
                if s_count!=0:
                    average[0].append(hr_static/s_count)
                if m_count!=0:
                    average[1].append(hr_move/m_count)  
                average[2].append(error_hr)

                p_test_hr = {"amb":{"all":[],"fb":[], "lr":[]},
                        "per":{"1m":[],"2m":[], "3m":[]},
                        "ran":{"1m":[],"2m":[], "3m":[]}}


                entry_m, entry_r = "", ""
                if 'noise' in path or 'place1' in path or 'place2' in path:
                    if "ambulant" in path:
                        entry_m = "amb"
                        entry_r = "all"
                    elif "periodical" in path:
                        entry_m = "per"
                        entry_r = "1m"  
                    elif "random" in path:
                        entry_m = "ran"    
                        entry_r = "1m"
                else: # px
                    if "ambulant" in path:
                        entry_m = "amb"
                        if "ambulant/all" in path:
                            entry_r = "all"
                        elif "ambulant/frontback" in path:
                            entry_r = "fb"
                        elif "ambulant/leftright" in path:
                            entry_r = "lr"
                    elif "periodical" in path:
                        entry_m = "per"
                        if "1m/periodical" in path:
                            entry_r = "1m"
                        elif "2m/periodical" in path:
                            entry_r = "2m"
                        elif "3m/periodical" in path:
                            entry_r = "3m"
                    elif "random" in path:
                        entry_m = "ran"    
                        if "1m/random" in path:
                            entry_r = "1m"
                        elif "2m/random" in path:
                            entry_r = "2m"
                        elif "3m/random" in path:
                            entry_r = "3m"

                p_test_hr[entry_m][entry_r].append(error_hr)
                # output excel
                
                output_result['Tester'].append(name_labels[i])
                output_result['motion'].append(entry_m)
                output_result['range'].append(entry_r)
                output_result['HR(GT)'].append(trans_list2str(val_targets[pi,:, 0]))
                output_result['HR(Pred)'].append(trans_list2str(val_predict[pi,:, 0]))
                output_result['static HR error'].append("{0:.2f}".format(static_error))
                output_result['move HR error'].append("{0:.2f}".format(move_error))
                output_result['HR error'].append("{0:.2f}".format(error_hr))
                output_result['First HR'].append("{0:.2f}".format(first_hr))
                output_result['Max HR'].append("{0:.2f}".format(max_hr))

                p_average_hr = (hr_all/val_predict.shape[0])         
                average[0].append(p_average_hr)
                model_list[test_name[0]] = model
                # print("Average {0}\n".format(p_average_hr))
                
                for m in p_test_hr:
                    for r in p_test_hr[m]:
                        data = p_test_hr[m][r]
                        if len(data) != 0:
                            sum_r = np.mean(np.array(data))
                            all_p_MSE_hr[m][r].append(sum_r)
            
            
            print("Output excel...\n")
            df = pd.DataFrame(output_result)
            writer = pd.ExcelWriter(EXCEL_NAME, engine='xlsxwriter')
            df.to_excel(writer, index=False)
            writer.save()
            print("---------------------------------\n")
    if STORE_EXCEL:
        {"amb":{"all":[],"fb":[], "lr":[]},
             "per":{"1m":[],"2m":[], "3m":[]},
             "ran":{"1m":[],"2m":[], "3m":[]}} 
        amb_all, amb_all_c = np.sum(all_p_MSE_hr["amb"]['all']), len(all_p_MSE_hr["amb"]['all'])
        amb_fb, amb_fb_c = np.sum(all_p_MSE_hr["amb"]['fb']), len(all_p_MSE_hr["amb"]['fb'])
        amb_lr, amb_lr_c = np.sum(all_p_MSE_hr["amb"]['lr']), len(all_p_MSE_hr["amb"]['lr'])

        per_1, per_1_c = np.sum(all_p_MSE_hr["per"]['1m']), len(all_p_MSE_hr["per"]['1m'])
        per_2, per_2_c = np.sum(all_p_MSE_hr["per"]['2m']), len(all_p_MSE_hr["per"]['2m'])
        per_3, per_3_c = np.sum(all_p_MSE_hr["per"]['3m']), len(all_p_MSE_hr["per"]['3m'])

        ran_1, ran_1_c = np.sum(all_p_MSE_hr["ran"]['1m']), len(all_p_MSE_hr["ran"]['1m'])
        ran_2, ran_2_c = np.sum(all_p_MSE_hr["ran"]['2m']), len(all_p_MSE_hr["ran"]['2m'])
        ran_3, ran_3_c = np.sum(all_p_MSE_hr["ran"]['3m']), len(all_p_MSE_hr["ran"]['3m'])

        output_result1 = {}
        output_result1["amb_range"] = ["all", "fb", "lr", "Avg"]
        
        output_result1["ambulant"] = ["{0:.2f}".format(amb_all/amb_all_c), 
                                      "{0:.2f}".format(amb_fb/amb_fb_c), 
                                      "{0:.2f}".format(amb_lr/amb_lr_c),
                                      "{0:.2f}".format((amb_all+amb_fb+amb_lr)/(amb_all_c+amb_fb_c+amb_lr_c))]
        
        output_result1['motion_range'] = ["1m", "2m", "3m", "Avg"]
        
        output_result1['perical'] = ["{0:.2f}".format(per_1/per_1_c), 
                                      "{0:.2f}".format(per_2/per_2_c), 
                                      "{0:.2f}".format(per_3/per_3_c),
                                      "{0:.2f}".format((per_1+per_2+per_3)/(per_1_c+per_2_c+per_3_c))]
        
        output_result1['random'] = ["{0:.2f}".format(ran_1/ran_1_c), 
                                      "{0:.2f}".format(ran_2/ran_2_c), 
                                      "{0:.2f}".format(ran_3/ran_3_c),
                                      "{0:.2f}".format((ran_1+ran_2+ran_3)/(ran_1_c+ran_2_c+ran_3_c))]
        
        output_result1['All'] = [" ", " ", " ", "{0:.2f}".format(np.mean(average[2]))]

        df = pd.DataFrame(output_result)
        df1 = pd.DataFrame(output_result1)
        
        writer = pd.ExcelWriter(EXCEL_NAME, engine='xlsxwriter')
        df1.to_excel(writer, sheet_name="Result",index=False)
        df.to_excel(writer, sheet_name="All",index=False)
        writer.close()

    if STROE_MODEL:
        for name in model_list:
            tmp_name = name.replace("/","")
            save_model(model_list[name], Save_model_folder, tmp_name)

    
    vital_cali(EXCEL_NAME)