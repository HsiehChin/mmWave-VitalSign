# Vital Signs Prediction under Free Body Movement based on Millimeter Wave Radar

By Ya-Fang Hsieh, Kai-Lung Hua


## Prerequisites
This code is written in Python 3.8 and requires the packages listed in requirements.txt. Install with pip install -r requirements.txt preferably in a virtualenv.

## Project Structure
```
mmWave-VitalSign/  
│  
├── CGU/  
│   └── src/  
│       ├── configs/  
│       ├── data_process/  
│       └── model/  
│  
└── datas/  
    ├── RobustVSDataset_anonymous/  
    └── cgu/  
```

## Datasets
- Baseline Dataset:  
The dataset for 'Rf vital sign sensing under free body movement' is self-collected. Detailed movement setup information is presented in the table below.  


|   Age  |                19-23 (4), 24-28 (7), 28-30 (3)                |
|:------:|:-------------------------------------------------------------:|
| Height |           1.60-1.70m (3), 1.70-1.80m (8), >1.80m (3)          |
| Gender |                     Male (13), Female (1)                     |
| Weight | 48-60kg (4), 60-70kg (3), 70-80kg (4), 80-90kg (2), >90kg (1) |


And related movement setup descriptions are presented in the table below.


| Motion Type |                      Subcategory                     |
|:-----------:|:----------------------------------------------------:|
|   Periodic  |                      1m, 2m, 3m                      |
|    Random   |                      1m, 2m, 3m                      |
|   Ambulant  | Front-back (fb), left-right (lr), comprehensive(com) |

- CGU Dataset:  
The CGU dataset includes 75 healthy participants. The related movement setup is presented in the table below.  

|   Motion Type   |       Subcategory      |
|:---------------:|:----------------------:|
|  Rope Skipping  | round1, round2, round3 |
| Stationary Bike | round1, round2, round3 |

In the CGU dataset, the training data includes two .csv files.  
- move-all.csv contains motion data from 75 participants. Its fields include Tester, Motion, GT(HR), Radar(HR), Power, Distance. The field descriptions are as follows:  

|   Header  |            Description            |
|:---------:|:---------------------------------:|
|   Tester  |             Tester ID             |
|   Motion  |           Type of motion          |
|   GT(HR)  |      Heart rate ground truth      |
| Radar(HR) |    Heart rate detected by radar   |
|   Power   |        Object motion power        |
|  Distance | Distance from radar to the object |

- fitness.csv contains the personalized fitness data of 75 participants. Its fields include Tester, Class, Score, and Gender. The field descriptions are as follows:

| Header |                                               Description                                               |
|:------:|:-------------------------------------------------------------------------------------------------------:|
| Tester |                                                Tester ID                                                |
|  Class | Fitness category: divided into "Low", "Moderate", "High",  corresponding to the numbers 1, 2, 3 respectively. |
|  Score |    Fitness score: a quantified score based on the amount of exercise in a week.  |
| Gender |                                           Participant's gender                                          |

## Run
### Step1. Download the mmWave-VitalSign project
- The source code in folder CGU/src
- The dataset in foler datas/, include Baseline dataset and CGU dataset
- You don't need to change any dataset structure.

### Step2. Change the content in {Dataset}_config.yaml
Direct to the CGU/src/configs folder

Both Baseline and CGU have their own config.yaml. In config.yaml, you can modify the relevant model parameters and output paths based on your needs.

- Baseline: baseline_config.yaml
- CGU: CGU_fitness_config.yaml

### Step3. Run the project
Navigate to the src folder and execute the command below. Once training is complete, the results will be saved in {The project path}/result/Baseline/ or {The project path}/result/CGU/.
Based on the parameters set in {Dataset}_config.yaml, the saved results include the following items:
  
{Dataset}/excel: Evaluation results  
{Dataset}/predict_result: Evaluation result images  
{Dataset}/saved_model: Trained weights  

- For Baseline train:  
  ```python baseline_train.py```

- For CGU train:  
  ```python CGU_fitness_train.py```


## System Evaluation
Evaluation metric : Mean Absolute Error (MAE)
Heart rate error unit: beats per minute (BPM)

- Baseline:
<p align="center">
  <img src="https://github.com/HsiehChin/mmWave-VitalSign/assets/49589536/116b49c6-0611-4e50-a47a-33dfb8ef4bb0" alt="Basseline result"/>
</p>
    
- CGU:  
<p align="center">
  <img src="https://github.com/HsiehChin/mmWave-VitalSign/assets/49589536/bc3a59e0-02d9-4ab3-bece-330bdf372bfc" alt="CGU result"/>
</p>

## Our Baseline
Jian Gong, Xinyu Zhang, Kaixin Lin, Ju Ren, Yaoxue Zhang, and Wenxun Qiu, “Rf vital sign sensing under free body movement,” Proceedings of the ACM on Inter- active, Mobile, Wearable and Ubiquitous Technologies, vol. 5, no. 3, pp. 1–22, 2021.


## Conference
Under review
