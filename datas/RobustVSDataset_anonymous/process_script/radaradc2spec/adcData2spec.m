clear
% load('0309range\dataDistance1_Raw_0.mat');
rootpath = ['D:\RobustVSDataset'];
middlepath = ['\xuanxi\fix\1m\periodical'];
% middlepath = ['\mayank\fix\1m\random'];
middlepath = ['\mayank\fix\2m\periodical'];
% middlepath = ['\mayank\fix\2m\random'];
% middlepath = ['\xuanxi\fix\3m\periodical'];
% middlepath = ['\mayank\fix\3m\random'];
% middlepath = ['\mayank\ambulant\leftright'];
% middlepath = ['\mayank\ambulant\frontback'];
% middlepath = ['\mayank\ambulant\all'];
datafolder = [rootpath, middlepath];

adcdatafolder = [datafolder, '\radar_adc\'];
adcdatafile = [adcdatafolder, 'radar_01.mat'];

load(adcdatafile);
% load('static-motion-bigmotion-static.mat');
rxNum = 1;%rx����������ȡ���������ļ�
numSamples = 512;
radar_para.sampleRate =10.24e6;
radar_para.freSlope = 68e12;
radar_para.framePeriod = 0.5e-3;
radar_para.frameRate = 1./radar_para.framePeriod;
%��λ=4��d/�������õ�d=��λ*����/4�У�77GHz����Ϊ4mm��������λ�仯1�൱�ھ���d�仯4mm/4��=0.3183mm
start_freq = 77e9;
c0 = physconst('LightSpeed');
wave_len = c0/start_freq;
range_phase_fac = wave_len/(4*pi)*1000;%Unit-mm
%range_phase_fac=0.3183;

attenaNum = size(adcData,1);
txNum = attenaNum./rxNum;%���tx�������ã�ȡ����ÿ��loop��chirp����
attenaArray = 1:attenaNum;
attenaArray = reshape(attenaArray, rxNum, txNum);
txIdx=1;%tx����ѡ�񣬶�Ӧ��chirp�����
rxIdx=1;%rx����ѡ�񣬶�Ӧ�ڰ��ص�8������֮һ
attenaIdx = attenaArray(rxIdx, txIdx);
loopsIdx = 1;
disp(size(adcData,4));
numFrames = size(adcData,4);


%%%%%%%%%%%%%%%%%%%%%
%--------ת����Ƶ��ͼ
%%%%%%%%%%%%%%%%%%%%%
for rxIdx = 1:1
    for txIdx = 1:1
        %% original IF data,observe if there is abnormal data
%         %--------��ĳ�������������ʱ�����������������һ����Ҫ�����Ҳ���������Ҫ���壩��
%         %ͬʱ������һЩ��Ƶ�Ӳ�������Զ����ı������壩
%         %--------������������壬���ǲ��������������Ҳ������Ҳ���˵���쳣
%         orig_frame = adcData(attenaIdx,loopsIdx,:,1);
%         figure
%         plot(real(squeeze(orig_frame)));
%         hold on
%         plot(imag(squeeze(orig_frame)));
%         hold off

        %% --------Range FFT
        attenaIdx = attenaArray(rxIdx, txIdx);
        zeropad_fac = 1;
        fft_length = numSamples*zeropad_fac;
        c0 = physconst('LightSpeed');
        range_axis = linspace(0,1-1/fft_length,fft_length)*radar_para.sampleRate*c0/(2*radar_para.freSlope);
        range_max = max(range_axis);
        range_perbin = range_max./numSamples;
        time_axis = [1:numFrames].*radar_para.framePeriod;
        fft_vecs = zeros(fft_length,numFrames);
        range_fft = zeros(fft_length,numFrames);
        %figure
        for frame_no = 1:numFrames
            temp_rf_frame = adcData(attenaIdx,loopsIdx,:,frame_no);
            temp_fft = fft(temp_rf_frame,fft_length);
            %temp_fft(near_range) = 0;
            % plot fft response
            %disp(size(temp_fft));
            %disp(temp_fft(1));
            %plot(range_axis,abs(real(temp_fft)))
            %drawnow
            range_fft(:,frame_no) = temp_fft;
        end
        %draw spectrum
        figure
        imagesc(time_axis,range_axis,abs(range_fft));
        %mesh(time_axis,range_axis,abs(range_fft));
        ylabel('range (meter)','FontSize',12);
        xlabel({'FFT Magnitude '},'FontSize',12);
        ylim([0 10]);
        
        %% subtract static objects
%         %
%         background_mean = mean(range_fft, 2);
%         range_fft = range_fft - background_mean;
%         figure
%         imagesc(time_axis,range_axis,abs(range_fft));
        
        %% ����������λ��-�൱��Ƶ��ͼ�����ľ���
        spec_mean = abs(range_fft);
        [argvalue, range_argmax] = max(spec_mean);
        bodyrange = range_argmax./numSamples.*range_max;
        [argvalue_mean, range_argmax_mean] = max(mean(abs(range_fft),2));
        range_argmax_mean = mean(range_argmax_mean);
        bodyrange_mean = range_argmax_mean./numSamples.*range_max;
%         figure
%         plot(time_axis, bodyrange)

        %% Movement Param
        W_all = zeros(numSamples, numFrames);
        P_all = zeros(numSamples, numFrames);
        range_all = zeros(numSamples, numFrames);
        %% Select range bin
        range_min = 0;
        range_max = 4;
        rangeIdxmin = max(1,round(range_min./range_perbin));
        rangeIdxmax = round(range_max./range_perbin);
%         for rangeIdx = max(0, range_argmax-0):min(range_argmax+0, numSamples)
        for rangeIdx = rangeIdxmin:rangeIdxmax
           %% original phase of selected range
            %phase = unwrap(angle(squeeze(adcData(attenaIdx,loopsIdx,rangeIdx,:))));
            phase = unwrap(angle(squeeze(range_fft(rangeIdx,:))));
            phase_origin = phase;
            phase_no_unwrap = angle(squeeze(range_fft(rangeIdx,:)));
            range = phase*range_phase_fac;
            delta_range = diff(range)./1000;%Unit-meter
            delta_range(numel(range)) = 0;%add one 0 to the end
            speed = diff(range)./radar_para.framePeriod./1000;
            speed(numel(range)) = 0;%add one 0 to the end
            accel = diff(speed)./radar_para.framePeriod;
            accel(numel(speed)) = 0;%add one 0 to the end
            F = 1*accel; %F=ma
            W = abs(F.*delta_range);%W=FS
            P= W./radar_para.framePeriod;%P=W/t
            
            W_all(rangeIdx,:) = W;
            P_all(rangeIdx,:) = P;
            range_all(rangeIdx,:) = range;
            
%              figure
%              plot(time_axis, range);
% %             hold on;
% %             plot(time_axis, phase_no_unwrap);
%             xlabel('time (s)','FontSize',12);
%             ylabel({'phase '},'FontSize',12);
            
%             %% plot phase-FFT
%             figure
%             fft_len = length(phase)*4;
%             phase_fft = fft(phase, fft_len);
%             freq_axis = linspace(0,1-1/fft_len,fft_len)*(1./radar_para.framePeriod*60);
%             plot(freq_axis, abs(phase_fft));
%             xlim([0 100]);
%             %ylim([0 2]);
%             xlabel('frequency (beats/seconds)','FontSize',12);
%             ylabel({'FFT Magnitude '},'FontSize',12);
            
%            %% apply high-pass filt.er
%             if 1
%                 hpFilt = designfilt('highpassfir','StopbandFrequency',0.8, ...
%                     'PassbandFrequency',1,'PassbandRipple',0.5, ...
%                     'StopbandAttenuation',20,'SampleRate',radar_para.frameRate,...
%                     'DesignMethod','kaiserwin');
%                 %                     fvtool(hpFilt)
%                 %                     D = mean(grpdelay(hpFilt)); % filter delay
%                 %                     phase_vec = filter(hpFilt,[phase_vec; zeros(D,1)]);
%                 %                     phase_vec = phase_vec(D+1:end);
%                 phase = filter(hpFilt,phase);
%             end
%             figure
%             plot(time_axis, phase);
%             xlabel('time (s)','FontSize',12);
%             ylabel({'phase '},'FontSize',12);
            
%             %% plot phase-FFT
%             figure
%             fft_len = length(phase)*4;
%             phase_fft = fft(phase, fft_len);
%             freq_axis = linspace(0,1-1/fft_len,fft_len)*(1./radar_para.framePeriod*60);
%             plot(freq_axis, abs(phase_fft));
%             xlim([0 100]);
%             %ylim([0 2]);
%             xlabel('frequency (beats/seconds)','FontSize',12);
%             ylabel({'FFT Magnitude '},'FontSize',12);
            
%             %% apply low-pass filter
%             if 1
%                 hpFilt = designfilt('lowpassfir','PassbandFrequency',1.8, ...
%                     'StopbandFrequency',2,'PassbandRipple',0.5, ...
%                     'StopbandAttenuation',20,'SampleRate',radar_para.frameRate,...
%                     'DesignMethod','kaiserwin');
%                 %                     fvtool(hpFilt)
%                 %                     D = mean(grpdelay(hpFilt)); % filter delay
%                 %                     phase_vec = filter(hpFilt,[phase_vec; zeros(D,1)]);
%                 %                     phase_vec = phase_vec(D+1:end);
%                 phase = filter(hpFilt,phase);
%             end
%             figure
%             plot(time_axis, phase);
%             xlabel('time (s)','FontSize',12);
%             ylabel({'phase '},'FontSize',12);
            
            
%            %% plot phase-FFT
%             figure
%             fft_len = length(phase)*4;
%             phase_fft = fft(phase, fft_len);
%             freq_axis = linspace(0,1-1/fft_len,fft_len)*(1./radar_para.framePeriod*60);
%             plot(freq_axis, abs(phase_fft));
%             xlim([0 100]);
%             %ylim([0 2]);
%             xlabel('frequency (beats/seconds)','FontSize',12);
%             ylabel({'FFT Magnitude '},'FontSize',12);
            
            
        end
        %% phase-range
%              figure
%              %plot(time_axis, range_all(1,:));
%              mesh(range_all);
% %             hold on;
% %             plot(time_axis, phase_no_unwrap);
% % %             hold on;
% % %             plot(time_axis, abs(accel));
%             xlabel('time (s)','FontSize',12);
%             ylabel({'phase '},'FontSize',12);

        %% sum up P
        P_instant = P_all.*abs(range_fft);
%         figure
%         mesh(time_axis, range_axis,P_instant)
        P_sum = sum(P_instant,1);
        figure
%         plot(time_axis, P_sum)
%         hold on;
       %% apply low-pass filter
        if 1
            hpFilt = designfilt('lowpassfir','PassbandFrequency',50, ...
                'StopbandFrequency',70,'PassbandRipple',0.5, ...
                'StopbandAttenuation',20,'SampleRate',radar_para.frameRate,...
                'DesignMethod','kaiserwin');
            P_sum = filter(hpFilt,P_sum);
        end
        plot(time_axis(1:end), P_sum(1:end),'LineWidth',1);
        xlabel('time (s)','FontSize',10); ylabel({'power'},'FontSize',10);
        set(gca,'XTick',[0:2:30]);set(gca,'YTick',[0:1:5]*1e8);%axis number
%         xlim([0,7]); 
        ylim([0,5].*1e8);%axis lim
        set(gca,'linewidth',1);% axis line width
        set(gca,'FontSize',10);%label font size
        x0=1;y0=1;width=3.3;height=2.3;set(gcf,'units','inches','position',[x0,y0,width,height]);%fig size
        
    end
end

