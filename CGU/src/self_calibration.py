import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_process.load_data import *

def get_gender():
    genders = """8:M
                 9:M
                10:M
                11:M
                12:F
                13:M
                14:M
                15:M
                16:M
                18:M
                20:M
                21:M
                22:M
                23:M
                24:M
                25:M
                26:M
                27:F
                28:M
                29:M
                30:F
                31:M
                32:M
                33:M
                34:M
                35:F
                36:M
                37:M
                38:M
                39:M
                41:M
                42:M
                43:F
                44:F
                45:M
                46:F
                47:M
                48:M
                49:M
                50:M
                51:M
                52:F
                53:F
                54:F
                55:M
                56:F
                57:M
                58:M
                59:M
                60:M
                61:M
                62:M
                63:M
                64:M
                65:M
                66:M
                67:F
                68:M
                69:F
                70:M
                71:M
                72:M
                73:M
                74:M
                75:M
                76:M
                77:M
                78:M
                79:M
                80:M
                81:M
                82:F
                83:M
                84:F"""
    output = {}
    genders = genders.strip().split('\n')
    for g in genders:
        tmp = g.strip().split(':')
        output[int(tmp[0])] = tmp[1]

    return output


# Get radar data from move-all.xlsx
# Need change the path to what you use
def get_radar_data(t, m):
    # All_data
    all_data_path = "/home/chin/Desktop/CGU/lilin-vital-sign-v0.9.6-cpp/recorded_data/export/All/move-all.csv"
    csv_data = pd.read_csv(all_data_path, engine='python')

    # Note,Tester,Motion,GT(HR),Radar(HR),Power,Distance
    output = [] 
    motions = csv_data['Motion']
    radars = csv_data['Radar(HR)']
    testers = csv_data['Tester']
    
    for i in range(len(testers)):
        if int(testers[i]) == int(t) and m.strip() == motions[i].strip():
            output = trans_str2list(radars[i])
            break
    
    return output


# Baseline self calibration
def vital_cali(excel_path):
    print(excel_path+": Output calibration...")
    output_name = excel_path.replace(".xlsx", "-cali.xlsx")
    df = pd.read_excel(excel_path, sheet_name="All")
    length = len(df['Tester'])

    new_df = {i:df[i] for i in df}
    # print(new_df)

    p = 0.4
    for i in range(length):
        ori_err = float(df['HR error'][i])

        hr_gt = np.array(trans_str2list(df['HR(GT)'][i]))
        hr_pred = np.array(trans_str2list(df['HR(Pred)'][i]))

        Dh = np.mean(hr_pred[:50]) - np.mean(hr_gt[:50])

        hr_pred = hr_pred - Dh*p
        
        error = np.sum(np.abs(hr_gt[:]-hr_pred[:]))/len(hr_gt[:])
        # print("Ori Error: {:.2f}, Error: {:.1f}".format(ori_err, error))

        # New setting
        new_df['HR(Pred)'][i] = trans_list2str(hr_pred)
        new_df['static HR error'][i] = np.mean(np.abs(hr_gt[:110]-hr_pred[:110]))
        new_df['move HR error'][i] = error
        new_df['HR error'][i] = error
        new_df['First HR'][i] = hr_pred[0]
        new_df['Max HR'][i] = np.max(hr_pred)


    err_all = np.array(new_df['HR error'], dtype=float)

    amb = [[], [], [], []] # All, all, fb, lr
    per = [[], [], [], []] # All, 1m, 2m, 3m
    ran = [[], [], [], []] # All, 1m, 2m, 3m

    for i in range(length):
        motion = df['motion'][i]
        range_str = df['range'][i]
        err = err_all[i]
        if "amb" in motion:
            amb[0].append(err)
            if "all" in range_str:
                amb[1].append(err)
            elif "fb" in range_str:
                amb[2].append(err)
            elif "lr" in range_str:
                amb[3].append(err)

        if "per" in motion:
            per[0].append(err)
            if "1m" in range_str:
                per[1].append(err)
            elif "2m" in range_str:
                per[2].append(err)
            elif "3m" in range_str:
                per[3].append(err)        

        if "ran" in motion:
            ran[0].append(err)
            if "1m" in range_str:
                ran[1].append(err)
            elif "2m" in range_str:
                ran[2].append(err)
            elif "3m" in range_str:
                ran[3].append(err)     


    output_result1 = {}
    output_result1["amb_range"] = ["all", "fb", "lr", "Avg"]

    output_result1["ambulant"] = ["{0:.2f}".format(np.mean(amb[1])), 
                                    "{0:.2f}".format(np.mean(amb[2])), 
                                    "{0:.2f}".format(np.mean(amb[3])),
                                    "{0:.2f}".format((np.mean(amb[0])))]

    output_result1['motion_range'] = ["1m", "2m", "3m", "Avg"]

    output_result1['perical'] = ["{0:.2f}".format(np.mean(per[1])), 
                                    "{0:.2f}".format(np.mean(per[2])), 
                                    "{0:.2f}".format(np.mean(per[3])),
                                    "{0:.2f}".format((np.mean(per[0])))]

    output_result1['random'] = ["{0:.2f}".format(np.mean(ran[1])), 
                                    "{0:.2f}".format(np.mean(ran[2])), 
                                    "{0:.2f}".format(np.mean(ran[3])),
                                    "{0:.2f}".format((np.mean(ran[0])))] 

    output_result1['All'] = [" ", " ", " ", "{0:.2f}".format(np.mean(err_all))]



    df = pd.DataFrame(new_df)
    df1 = pd.DataFrame(output_result1)
    writer = pd.ExcelWriter(output_name, engine='xlsxwriter')
    df1.to_excel(writer, sheet_name="Result", index=False)
    df.to_excel(writer, sheet_name="All", index=False)
    writer.save()


# CGU self calibration, then output new excel
def CGU_cali(excel_path, cali_flag=0):
    print(excel_path+": Output calibration...")
    
    output_name = excel_path.replace(".xlsx", "-cali.xlsx")
    
    df = pd.read_excel(excel_path, sheet_name="All")
    length = len(df['Tester'])-1

    err_all = np.array(df['HR error'], dtype=float)
    ride = [] 
    jump = []

    # genders = get_gender()
    # static_hr = [[], [], [], []] # ori, cali, male. female

    new_df = {i:df[i] for i in df}

    p = 0.4
    for i in range(length):
        tester = df['Tester'][i]
        motion = df['motion'][i]

        radar_hr = get_radar_data(tester, motion)

        hr_gt = np.array(trans_str2list(df['HR(GT)'][i]))

        if cali_flag == 0:
            cali_hr = hr_gt
            p=0.4
        else:
            if '1' in motion:
                cali_hr = np.array(radar_hr)
                p = 0.4
            else:
                p = 0.8
                cali_hr = hr_gt
                
        hr_pred = np.array(trans_str2list(df['HR(Pred)'][i]))

        flag = get_flag(tester, motion)
        ema_hr = HR_EMA(hr_pred, flag, type=1)
        hr_pred = np.array(ema_hr)

        Dh = np.mean(hr_pred[40:60]) - np.mean(cali_hr[40:60])
    

        cali_hr_pred = hr_pred - Dh*p
        ori_error_hr = np.abs(hr_gt-hr_pred)       
        error_hr = np.abs(hr_gt-cali_hr_pred)


        if int(tester) < 40:
            # front_error_hr = error_hr[:180]
            # end_error_hr = error_hr[60:]

            # move_error = (np.sum(front_error_hr)/180+np.sum(end_error_hr)/60)/2
            move_error = np.sum(error_hr)/len(error_hr)

        else:
            if "jump" in motion or "Jump" in motion:
                # front_error_hr = error_hr[:120]
                # end_error_hr = error_hr[:]
                move_error = np.sum(error_hr)/len(error_hr)

                # move_error = (np.sum(error_hr[:120])/120+np.sum(error_hr[120:180])/60+np.sum(error_hr[180:240])/60)/3
            else:    
                # front_error_hr = error_hr[:60]
                # end_error_hr = error_hr[:]
                # move_error, m_min, m_max = np.mean(end_error_hr), np.min(end_error_hr), np.max(end_error_hr)
                move_error = np.sum(error_hr)/len(error_hr)
        # print("Ori Error: {:.2f}, Error: {:.1f}".format(ori_err, error))

        # New setting
        new_df['HR(Pred)'][i] = trans_list2str(cali_hr_pred)
        new_df['move HR error'][i] = move_error
        new_df['HR error'][i] = move_error
        new_df['First HR'][i] = cali_hr_pred[0]
        new_df['Max HR'][i] = np.max(cali_hr_pred)

    # -----------------
    err_all = np.array(new_df['HR error'], dtype=float)
    err_m_all = np.array(new_df['move HR error'], dtype=float)

    ride = [] 
    jump = []
    """
    Static	Move	Average	Jump error	Ride error
    """
    
    for i in range(length):
        motion = df['motion'][i]
        err_m = err_all[i]
        if "jump" in motion or "Jump" in motion:
            jump.append(err_m)
        else:
            ride.append(err_m)

        df = pd.DataFrame(new_df)
        df1 = pd.DataFrame({
                            "Move":	        ["{0:.2f}".format(np.sum(err_m_all)/len(err_m_all))],
                            "Average":	    ["{0:.2f}".format(np.sum(err_all)/len(err_all))],
                            "Jump error":	["{0:.2f}".format(np.sum(jump)/len(jump))],
                            "Ride error":	["{0:.2f}".format(np.sum(ride)/len(ride))],
                            "Euclidean distance": ["{0:.2f}".format(np.sum(eu_ds)/len(eu_ds))],
                            "Correlation coefficient": ["{0:.2f}".format(np.sum(ccs)/len(ccs))],
                            "DTW distance": ["{0:.2f}".format(np.sum(dds)/len(dds))],
                            })
        
        writer = pd.ExcelWriter(output_name, engine='xlsxwriter')
        df1.to_excel(writer, sheet_name="Result",index=False)
        df.to_excel(writer, sheet_name="All",index=False)
        writer.save()

    # print("For 實驗用：", len(static_hr[0]))
    # print("Static Error: ", sum(static_hr[0])/len(static_hr[0]))
    # print("Static Cali Error: ", sum(static_hr[1])/len(static_hr[1]))
    # print("Male Cali Error: ", sum(static_hr[2])/len(static_hr[2]))
    # print("Female Cali Error: ", sum(static_hr[3])/len(static_hr[3]))
    # print(static_hr[0])
    # print(static_hr[1])


# Split the CGU predicted excel into 3 rounds.
def CGU_split_round(excel_path):
    print(excel_path+": Output calibration...")
    output_name = excel_path.replace(".xlsx", "-round.xlsx")
    
    df = pd.read_excel(excel_path, sheet_name="All")
    length = len(df['Tester'])-1


    err_all = np.array(df['HR error'], dtype=float)
    ride = [[], [], []] 
    jump = [[], [], []]
    eu_ds = []
    ccs = [] 
    dds = [] 

    genders = get_gender()
    static_hr = [[], [], [], []] # ori, cali, male. female

    new_df = {'Jump 1':[], 'Jump 2':[], 'Jump 3':[], 'Jump Average':[],
              'Ride 1':[], 'Ride 2':[], 'Ride 3':[], 'Ride Average':[]
              }

    p = 0.4
    for i in range(length):
        tester = df['Tester'][i]
        motion = df['motion'][i]
        err = df['HR error'][i]

        if 'jump' in motion.lower():
            if '1' in motion:
                jump[0].append(err)
            elif '2' in motion:
                jump[1].append(err)
            elif '3' in motion:
                jump[2].append(err)
        elif 'ride' in motion.lower():
            if '1' in motion:
                ride[0].append(err)
            elif '2' in motion:
                ride[1].append(err)
            elif '3' in motion:
                ride[2].append(err)

    # New setting
    new_df['Jump 1'].append(sum(jump[0])/len(jump[0]))
    new_df['Jump 2'].append(sum(jump[1])/len(jump[1]))
    new_df['Jump 3'].append(sum(jump[2])/len(jump[2]))

    jump_err = jump[0]
    jump_err += jump[1]
    jump_err += jump[2]

    new_df['Jump Average'].append(sum(jump_err)/len(jump_err))

    new_df['Ride 1'].append(sum(ride[0])/len(ride[0]))
    new_df['Ride 2'].append(sum(ride[1])/len(ride[1]))
    new_df['Ride 3'].append(sum(ride[2])/len(ride[2]))

    ride_err = ride[0]
    ride_err += ride[1]
    ride_err += ride[2]

    new_df['Ride Average'].append(sum(ride_err)/len(ride_err))


    df = pd.DataFrame(new_df)    
    writer = pd.ExcelWriter(output_name, engine='xlsxwriter')
    df.to_excel(writer, sheet_name="Result",index=False)
    writer.save()    