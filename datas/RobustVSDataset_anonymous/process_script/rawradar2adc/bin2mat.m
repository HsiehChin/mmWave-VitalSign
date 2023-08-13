function re = bin2mat(input,output,radar_para,sel_ant_flag,sel_ant_no,down_sample_flag,capon_beamform_flag,simple_beamform_flag) 

for ii = 1:length(input)     
    %% source/destine file path search complete, the process begin
    num_loop = radar_para.num_loop;
    period = radar_para.framePeriod;
    num_sample = radar_para.num_sample;
    num_tx = radar_para.num_tx;
    num_rx = radar_para.num_rx;
    num_ant = num_tx * num_rx;

    %% Load RF data
    input{ii}
    [raw_data] = readDCA1000(input{ii});
    size(raw_data)
    num_frame = size(raw_data, 2) /num_sample/num_tx/num_loop;
    num_frame
    adcData = zeros(num_sample, num_ant, num_loop, num_frame);
    for mm = 1:num_frame
        for nn = 1:num_loop
            for kk = 1:num_tx
                for ll = 1:num_rx
                    adcData(:,(kk-1)*num_rx+ll,nn,mm) = raw_data(ll,((mm-1)*num_loop*num_tx+(nn-1)*num_tx+kk-1)*num_sample+(1:num_sample));
                end
            end
        end
    end
   
    adcData = (adcData);
    adcData = permute(adcData,[2,3,1,4]);
    size(adcData)
    %% select only one Rx channel
    if sel_ant_flag
        dataAll_wo_bf = adcData(:,sel_ant_no,:);
    end



    %% capon beamform
    if capon_beamform_flag
        ula = phased.ULA('NumElements',8,'ElementSpacing',radar_para.lambda/2);
        ula.Element.FrequencyRange = [77e9 81e9];
        caponBF_gen = phased.MVDRBeamformer('SensorArray',ula,...
            'OperatingFrequency',radar_para.fc0 ,'DirectionSource','Property',...
            'Direction',[0;0],'WeightsOutputPort',true);


        %                     caponBF_gen = phased.SubbandMVDRBeamformer('SensorArray',ula,...
        %                         'OperatingFrequency',radar_para.fc0 ,...
        %                         'SampleRate',radar_para.sampleRate,...
        %                         'NumSubbands',2,...
        %                         'DirectionSource','Property',...
        %                         'Direction',[0;0],...
        %                         'WeightsOutputPort',true);

        BFed_dataAll = NaN([num_dp,1,num_frame],'double');


        figure
        for iii = 1:num_frame
            [BFed_data,BFed_weight] = caponBF_gen(adcData(:,:,iii));
            plot(abs(BFed_weight))
            hold on
            drawnow
            BFed_dataAll(:,1,iii) = BFed_data;
        end
        adcData = BFed_dataAll;
        clear BFed_dataAll
        clear BFed_data
    end


    %% simple beamforming
    if simple_beamform_flag
        dataAll_w_bf = mean(adcData,2);
    end


    %% down sample, that is, take average every 4 sample
    num_dp = num_sample;
    if down_sample_flag
        num_dp = num_sample/4;
        temp_dataAll = zeros(num_dp,1,num_frame);
        if simple_beamform_flag
            for frame_no = 1:num_frame
                for kk = 1:num_dp
                    temp_dataAll(kk,:,frame_no) = mean(dataAll_w_bf(1+(kk-1)*4:1+kk*4-1,:,frame_no),1);
                end
            end
            dataAll_w_bf = temp_dataAll;
        end
        if sel_ant_flag
            for frame_no = 1:num_frame
                for kk = 1:num_dp
                    temp_dataAll(kk,:,frame_no) = mean(dataAll_wo_bf(1+(kk-1)*4:1+kk*4-1,:,frame_no),1);
                end
            end
            dataAll_wo_bf = temp_dataAll;
        end
        temp_dataAll = zeros(num_dp,8,num_frame);
        if ~sel_ant_flag
            for frame_no = 1:num_frame
                for antena_num = 1:8
                    for kk = 1:num_dp
                        temp_dataAll(kk,antena_num,frame_no) = mean(adcData(1+(kk-1)*4:1+kk*4-1,antena_num,frame_no),1);
                    end
                end
            end
            adcData = temp_dataAll;
        end
    end
    
    
    %% save to file
    save(output{ii},'adcData');
end