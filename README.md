# Vital Signs Prediction under Free Body Movement based on Millimeter Wave Radar

By Ya-Fang Hsieh, Kai-Lung Hua


## Prerequisites
This code is written in Python 3.8 and requires the packages listed in requirements.txt. Install with pip install -r requirements.txt preferably in a virtualenv.

## Project Structure
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


## Run
### Step1. Download the mmWave-VitalSign project
- The source code in folder CGU/src
- The dataset in foler datas/, include Baseline dataset and CGU dataset
- You don't need to change any dataset structure.

### Step2. Change the ...config.yaml
Direct to the CGU/src/configs folder

Both Baseline and CGU have their own config.yaml. In config.yaml, you can modify the relevant model parameters and output paths based on your needs.

- Baseline: baseline_config.yaml
- CGU: CGU_fitness_config.yaml

The training results will be generated in the CGU/result folder.

### Step3. Run the project
Direct to the src folder
- For Baseline train:
    python baseline_train.py

- For CGU train:
    python CGU_fitness_train.py


## Our Baseline
Jian Gong, Xinyu Zhang, Kaixin Lin, Ju Ren, Yaoxue Zhang, and Wenxun Qiu, “Rf vital sign sensing under free body movement,” Proceedings of the ACM on Inter- active, Mobile, Wearable and Ubiquitous Technologies, vol. 5, no. 3, pp. 1–22, 2021.


## Conference
Under review