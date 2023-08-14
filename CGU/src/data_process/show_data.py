import os, yaml
import numpy as np
import matplotlib.pyplot as plt
from data_process.load_data import readDir

# draw_graph for baseline
def draw_graph(pathArray, result_folder, power, ground_truth, predict, mode = 0, yaml_path='config.yaml'):
    
    if not os.path.exists(result_folder):
        print("make dir")
        os.makedirs(result_folder)

    with open(yaml_path) as stream:
        try:
            config = yaml.full_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    DATA_FOLDER_PATH = os.path.abspath(os.path.join(os.getcwd(), config['DATA_FOLDER_PATH'])) 
    # print("Power: ", power.shape)
    # print("ground_truth: ", ground_truth.shape)
    # print("predict: ", predict.shape)
    tmp_path_array = []
    for i in range(len(pathArray)):
        tmp_path = pathArray[i]
        tmp_path = tmp_path.replace(DATA_FOLDER_PATH, "")
        tmp_list = tmp_path.split("/")
        tmp_list.pop(-2)
        tmp_path = "_".join(tmp_list)
        # tmp_path = (name + tmp_path).replace("/","_")
        tmp_path_array.append(tmp_path)


    # Draw graph
    time_length = power.shape[1]
    for i in range(len(tmp_path_array)):
        # print(i+1, tmp_path_array[i])
        # x: time axis, y: power, hr, rr
        
        # draw power, hr, rr
        power_i = power[i,:,0]
        if power.shape[2] >= 3:
            distance_i = power[i, :, 1]
            move_weight_i = power[i, :, 2]
        elif power.shape[2] >= 2:
            distance_i = power[i, :, 1]
            
        hr_i = ground_truth[i,:,0] # hr
        # rr_i = ground_truth[i,:,1] # rr
        pre_hr_i = predict[i,:,0] # predict hr
        # pre_rr_i = predict[i,:,1] # predict rr
        
        # plotting the line hr points 
        plt.figure(figsize=(15, 15))

        # plt.subplot(211)
        plt.subplot(411)
        plt.plot(hr_i, label = "HR", color="red")
        plt.plot(pre_hr_i, label = "pre HR", color="orange")
        plt.xlabel('time')
        plt.ylabel('Heart rate')
        plt.title("heart rate", fontsize=8, color='red')   
        plt.legend()
        
        plt.subplot(412)
        plt.plot(power_i, label = "Power", color="blue")
        plt.xlabel('time')
        plt.title("power", fontsize=8, color='blue')   
        plt.legend()
        plt.tight_layout()
        if power.shape[2] >= 2:
            plt.subplot(413)
            plt.plot(distance_i, label = "Distance", color="blue")
            plt.xlabel('time')
            plt.title("Distance", fontsize=8, color='blue')   
            plt.legend()
            plt.tight_layout()

        if power.shape[2] >= 3:
            plt.subplot(414)
            plt.plot(move_weight_i, label = "Move weight", color="blue")
            plt.xlabel('time')
            plt.title("Move weight", fontsize=8, color='blue')   
            plt.legend()
            plt.tight_layout()

        # plotting the line rr points 
        # plt.subplot(412)
        # plt.plot(rr_i, label = "RR", color="green")
        # plt.plot(pre_rr_i, label = "pre RR", color="orange")
        # plt.xlabel('time')
        # plt.ylabel('Breath rate')
        # plt.title("breath rate", fontsize=8, color='red')   
        # plt.legend()

        # function to show the plot
        plt.savefig("{0}/{1}.png".format(result_folder, tmp_path_array[i]))
        plt.close('all')


# draw CGU graph
def draw_graph_CGU_fitness(pathArray, power_ori, power, ground_truth, predict, yaml_path='config_CGU_fitness.yaml'):
    
    with open(yaml_path) as stream:
        try:
            config = yaml.full_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    DATA_FOLDER_PATH = os.path.abspath(config['DATA_FOLDER_PATH']) 
    Save_result_folder = os.path.abspath(config['Save_result_folder'])
    

    if not os.path.exists(Save_result_folder):
        print("make dir")
        os.makedirs(Save_result_folder)

    # Draw graph, pathArray means number of motion of the tester
    if len(pathArray) > 3:
        if len(pathArray) == 4:
            path_list = [[0, 1], [2, 3]]
        elif len(pathArray) == 5:
            path_list = [[0, 1, 2], [3, 4]]
        elif len(pathArray) == 6:
            path_list = [[0, 1, 2], [3, 4, 5]]

        plt.figure(figsize=(15,15))
        
        for ni, i in enumerate(path_list):
            col = len(i)
            min_index = min(i)
            for k in i:
                # x: time axis, y: power, hr, rr
                # draw power, hr, rr
                # input with normalization
                feature_size = power.shape[2]
                power_weight_i = power[k,:,0]
                if feature_size >= 6:
                    distance_i = power[k,:,1]
                    move_time_i = power[k,:,2]
                elif feature_size == 5:
                    distance_i = power[k,:,1]
                    move_time_i = power[k,:,2]
                elif feature_size == 4:
                    move_time_i = power[k,:,1]
                    distance_i = np.array(move_time_i)

                # input no normalization
                if len(power_ori.shape) == 4:
                    power_i = power_ori[k,:,0]
                    radar_hr_ori_i = power_ori[k,:,2]

                hr_i = ground_truth[k,:,0] # hr
                # rr_i = ground_truth[k,:,1] # rr
                pre_i = predict[k,:,0] # predict hr

                # plotting the line hr
                
                plt.subplot(4, col, (k+1)-min_index)

                plt.plot(hr_i, label = "HR", color="red")
                # plt.plot(radar_hr_ori_i, label = "radar HR", color="green")
                plt.plot(pre_i, label = "predicted HR", color="orange")
                plt.ylabel('Heart rate')
                plt.xlabel('time')
                plt.legend()
                plt.title('{0}'.format(k+1), fontsize=8) # plot title

                # plotting the line power
                plt.subplot(4, col, (k+1+col)-min_index)
                
                plt.plot(power_weight_i, label = "power", color="blue")
                # plt.plot(distance_i, label = "distance", color="gray")
                plt.ylabel('Power(W)')
                plt.xlabel('time')
                plt.legend()
                plt.title('{0} power weighted'.format(k+1), fontsize=8) # plot title

                # plotting the line power
                plt.subplot(4, col, (k+1+2*col)-min_index)
                
                plt.plot(move_time_i, label = "Move time weighted", color="blue")
                # plt.plot(distance_i, label = "distance", color="gray")
                plt.ylabel('mvoe times')
                plt.xlabel('time')
                # show a legend on the plot
                plt.title('{0} move time weight'.format(k+1), fontsize=8) # plot title

                # plotting the line power
                plt.subplot(4, col, (k+1+3*col)-min_index)                
                plt.plot(distance_i, label = "Distance", color="blue")
                # plt.plot(distance_i, label = "distance", color="gray")
                plt.ylabel('distance(m)')
                plt.xlabel('time')
                # show a legend on the plot
                plt.title('{0} Distance'.format(k+1), fontsize=8) # plot title

                plt.legend()

            # plt.show()
            plt.tight_layout()
            name = pathArray[0]
            
            plt.savefig("{0}/{1}-{2}.png".format(Save_result_folder, name, ni+1))
            plt.clf()
        plt.close('all')

    elif len(pathArray) > 1:
        plt.figure(figsize=(15,15))
        col = len(pathArray)
        for i in range(len(pathArray)):
            # x: time axis, y: power, hr, rr
            # draw power, hr, rr

            # input with normalization
            feature_size = power.shape[2]
            power_weight_i = power[i,:,0]
            if feature_size >= 6:
                distance_i = power[i,:,1]
                move_time_i = power[i,:,2]
            elif feature_size == 5:
                distance_i = power[i,:,1]
                move_time_i = power[i,:,2]
            elif feature_size == 4:
                move_time_i = power[i,:,1]
                distance_i = np.array(move_time_i)

            # input no normalization
            if power_ori.shape[-1] == 4:
                power_i = power_ori[i,:,0]
                radar_hr_ori_i = power_ori[i,:,2]

            hr_i = ground_truth[i,:,0] # hr
            # rr_i = ground_truth[i,:,1] # rr
            pre_i = predict[i,:,0] # predict hr

            # plotting the line hr
            plt.subplot(4, col, i+1)
            plt.plot(hr_i, label = "HR", color="red")
            # plt.plot(radar_hr_ori_i, label = "radar HR", color="green")
            plt.plot(pre_i, label = "predicted HR", color="orange")
            plt.ylabel('Heart rate')
            plt.xlabel('time')
            plt.legend()
            plt.title('{0}'.format(i+1), fontsize=8) # plot title

            # plotting the line power
            plt.subplot(4, col, i+1+col)
            plt.plot(power_weight_i, label = "power", color="blue")
            # plt.plot(distance_i, label = "distance", color="gray")
            plt.ylabel('Power(W)')
            plt.xlabel('time')
            plt.legend()
            plt.title('{0} power weighted'.format(i+1), fontsize=8) # plot title

            # plotting the line power
            plt.subplot(4, col, i+1+2*col)
            plt.plot(move_time_i, label = "Move time weighted", color="blue")
            # plt.plot(distance_i, label = "distance", color="gray")
            plt.ylabel('mvoe times')
            plt.xlabel('time')
            # show a legend on the plot
            plt.title('{0} move time weight'.format(i+1), fontsize=8) # plot title

            # plotting the line power
            plt.subplot(4, col, i+1+3*col)
            plt.plot(distance_i, label = "Distance", color="blue")
            # plt.plot(distance_i, label = "distance", color="gray")
            plt.ylabel('distance(m)')
            plt.xlabel('time')
            # show a legend on the plot
            plt.title('{0} Distance'.format(i+1), fontsize=8) # plot title

            plt.legend()

        # plt.show()
        plt.tight_layout()
        name = pathArray[0]
        
        plt.savefig("{0}/{1}.png".format(Save_result_folder, name))
        plt.close('all')

    elif len(pathArray) == 1:
        plt.figure(figsize=(15,15))
        for i in range(len(pathArray)):
            # x: time axis, y: power, hr, rr
            # draw power, hr, rr

            # input with normalization
            feature_size = power.shape[2]
            power_weight_i = power[i,:,0]
            if feature_size >= 6:
                distance_i = power[i,:,1]
                move_time_i = power[i,:,2]
            elif feature_size == 5:
                distance_i = power[i,:,1]
                move_time_i = power[i,:,2]
            elif feature_size == 4:
                move_time_i = power[i,:,1]
                distance_i = np.array(move_time_i)

            # input no normalization
            if power_ori.shape[-1] == 4:
                power_i = power_ori[i,:,0]
                radar_hr_ori_i = power_ori[i,:,2]

            hr_i = ground_truth[i,:,0] # hr
            # rr_i = ground_truth[i,:,1] # rr
            pre_i = predict[i,:,0] # predict hr

            # plotting the line hr
            plt.subplot(4, 1, 1)
            plt.plot(hr_i, label = "HR", color="red")
            # plt.plot(radar_hr_ori_i, label = "radar HR", color="green")
            plt.plot(pre_i, label = "predicted HR", color="orange")
            plt.ylabel('Heart rate')
            plt.xlabel('time')
            plt.legend()
            plt.title('{0}'.format(i+1), fontsize=8) # plot title

            # plotting the line power
            plt.subplot(4, 1, 2)
            plt.plot(power_weight_i, label = "power", color="blue")
            # plt.plot(distance_i, label = "distance", color="gray")
            plt.ylabel('Power(W)')
            plt.xlabel('time')
            plt.legend()
            plt.title('{0} power weighted'.format(i+1), fontsize=8) # plot title

            # plotting the line power
            plt.subplot(4, 1, 3)
            plt.plot(move_time_i, label = "Move time weighted", color="blue")
            # plt.plot(distance_i, label = "distance", color="gray")
            plt.ylabel('mvoe times')
            plt.xlabel('time')
            # show a legend on the plot
            plt.title('{0} move time weight'.format(i+1), fontsize=8) # plot title

            # plotting the line power
            plt.subplot(4, 3, 4)
            plt.plot(distance_i, label = "Distance", color="blue")
            # plt.plot(distance_i, label = "distance", color="gray")
            plt.ylabel('distance(m)')
            plt.xlabel('time')
            # show a legend on the plot
            plt.title('{0} Distance'.format(i+1), fontsize=8) # plot title

            plt.legend()


        # plt.show()
        plt.tight_layout()
        name = pathArray[0]
        
        plt.savefig("{0}/{1}.png".format(Save_result_folder, name))
        plt.close('all')


# Draw Loss graph
def draw_loss_graph(name, train_loss, val_loss, yaml_path='config_CGU.yaml'):

    with open(yaml_path) as stream:
        try:
            config = yaml.full_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    path = config['Save_loss_folder']

    # Creating the plot
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')

    # Adding labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')

    # Adding a legend
    plt.legend()

    # Saving the plot as an image file
    plt.savefig(path+"_{}.png".format(name))


# Draw feature graph
def draw_hidden_graph(name, predict_hidden, yaml_path='config_CGU_contextual.yaml'):

    with open(yaml_path) as stream:
        try:
            config = yaml.full_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    Save_result_folder = os.path.abspath(config['Save_result_folder'])+"/hidden"
    

    if not os.path.exists(Save_result_folder):
        print("make dir")
        os.makedirs(Save_result_folder)

    row = 0
    col = 0
    show = {}
    for h in predict_hidden:
        if predict_hidden[h].shape[2]==1:
            row+=1
            col = predict_hidden[h].shape[1]
            show[h] = predict_hidden[h].cpu().detach().numpy() 

    # Draw graph
    ri = 1
    for h in show:
        data = show[h]
        for ci in range(data.shape[1]):
            # plotting the line hr
            plt.subplot(row, col, ri+ci)
            plt.plot(data[:, ci, 0], label = h, color="blue")
            # plt.ylabel(h)
            # plt.xlabel('time')
            plt.title('{0}'.format(h), fontsize=8) # plot title            
        
        ri += data.shape[1]

    # plt.show()
    plt.tight_layout()    
    plt.savefig("{0}/{1}.png".format(Save_result_folder, name))
    plt.close('all')
