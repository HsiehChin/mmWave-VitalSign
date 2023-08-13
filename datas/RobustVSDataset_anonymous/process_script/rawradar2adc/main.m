clear;
%% chirp parameters
c0 = physconst('LightSpeed');
radar_para.sampleRate =1e7;
radar_para.freSlope = 29.9817e12;
radar_para.framePeriod = 40e-3;
radar_para.frameRate = 1/radar_para.framePeriod;
radar_para.fc0 = 77e9;
RampEndTime = 60;
ADCStartTime = 6;
radar_para.fc0 = 77e9+4e9*(ADCStartTime/RampEndTime);
radar_para.lambda = c0/radar_para.fc0;
radar_para.num_loop = 1;
radar_para.num_tx = 1;
radar_para.num_rx = 1;
radar_para.num_sample = 512;

currentFolder = pwd;
rootpath = ['D:\RobustVSDataset'];
middlepath = ['\ish\fix\1m\periodical'];
% middlepath = ['\ish\fix\1m\random'];
middlepath = ['\xuanxi\fix\2m\periodical'];
% middlepath = ['\xuanxi\fix\2m\random'];
middlepath = ['\xuanxi\fix\3m\periodical'];
% middlepath = ['\xuanxi\fix\3m\random'];
% middlepath = ['\xuanxi\ambulant\leftright'];
% middlepath = ['\xuanxi\ambulant\frontback'];
% middlepath = ['\xuanxi\ambulant\all'];
datafolder = [rootpath, middlepath];

rawdatafolder = [datafolder, '\rawdata\'];
adcdatafolder = [datafolder, '\radar_adc\'];
if ~exist(adcdatafolder, 'dir')
  mkdir(adcdatafolder);
end

% folderpath = [currentFolder '\0317kun\'];
% folderpath = [currentFolder '\0720test\'];
bin_path = rawdatafolder;
mat_path = adcdatafolder;

num = [1:100];
input = cell(1,length(num));
output = input;
for i = 1:length(num)
%     input{i} = [bin_path,'radar_','',int2str(num(i)),'_Raw_0','.bin'];
%     output{i} = [mat_path,'radar_',int2str(num(i)),'_Raw_0','.mat'];
%     bin_file = [bin_path,'radar_',num2str(num(i),'%02d'),'_Raw_0','.bin'];
%     mat_file = [mat_path,'radar_',num2str(num(i),'%02d'),'.mat'];
    input{i} = [bin_path,'radar_',num2str(num(i),'%02d'),'_Raw_0','.bin'];
    output{i} = [mat_path,'radar_',num2str(num(i),'%02d'),'.mat'];
    if ~isfile(input{i})
         break;
    end
    
end
% filename = ['group12_Raw_0'];
% input{1} = [bin_path,filename,'.bin'];
% output{1} = [mat_path,filename,'.mat'];


sel_ant_flag = 0; sel_ant_no = 3;
down_sample_flag = 0;
capon_beamform_flag = 0;
simple_beamform_flag = 0;
bin2mat(input,output,radar_para,sel_ant_flag,sel_ant_no,down_sample_flag,capon_beamform_flag,simple_beamform_flag);
% if sel_ant_flag
%     radar_para.num_tx = 1;
%     radar_para.num_rx = 1;
% end
% if capon_beamform_flag
%     radar_para.num_tx = 1;
%     radar_para.num_rx = 1;
% end
% if down_sample_flag
%     radar_para.num_sample = radar_para.num_sample/4;
%     radar_para.sampleRate = radar_para.sampleRate/4;
% end
% raw_rf2range_vec(data_path,radar_para);

