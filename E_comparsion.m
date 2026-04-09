% compare the following three methods:
% 1. proposed method
% 2. data-driven method --- input: voltage, current and OCV
% 4. model-driven method --- EKF
% also compare the different start point
% 1. 90%
% 2. 60%


clear
clc
warning off

SEED_INDEX = 1;
TEMP_GROUP = {'N10', '0', '10', '20', '30', '40', '50'};
TEMP_ERROR_GROUP = TEMP_GROUP;

CONDITION_GROUP = {'DST', 'FUDS', 'US06'};
WINDOW_GROUP = 100:100:500;

rng(SEED_INDEX)

cond = 'DST';
temp = '20';

windsize = 300;

save_fname = strcat('.\resultsStorage\compare_results_different_inputs_and_EKF.mat');

fname = strcat('.\processedData\data_t_',temp,'_idx_1.mat');
load(fname)

soc_ini = 0.8;

result_all = [];
for start_point = [0.9, 0.6]
    for method_flag = 1:2
        [result_all.(strcat('s_', num2str(start_point*100))).(strcat('m_', num2str(method_flag))).train_result, ...
            result_all.(strcat('s_', num2str(start_point*100))).(strcat('m_', num2str(method_flag))).result] = ...
            func_base_error(soc_ini, start_point, method_flag, data, TEMP_ERROR_GROUP, TEMP_GROUP, CONDITION_GROUP);
    end

    %% EKF
    mae_no_KF_all = [];
    rmse_no_KF_all = [];
    % for temp_index = 1:7
    for temp_index = [1, 4, 7]
        temp = TEMP_GROUP{temp_index};
        fname = strcat('.\processedData\data_t_',temp,'_idx_1.mat');
        load(fname)
        for cond_index = 2:3
            cond = CONDITION_GROUP{cond_index};
            SOC = data.(cond).extracted_data(:, 3);
            sp = find(SOC <= start_point, 1);

            I = -data.(cond).extracted_data(sp:end, 1);
            Ut = data.(cond).extracted_data(sp:end, 2);

            N=size(Ut,1); %矩阵长度为N

            Cn = data.(cond).extracted_data(end, 4) -  data.(cond).extracted_data(1, 4);%电池可用能量
            cap_rec_1 = (1+(4*rand-2)/100) ;
            Cn = cap_rec_1* Cn;
            Z=zeros(2,N);
            Z(:,1)=[0, soc_ini]; %[Up SOC]
            Qz=diag([1e-6,1e-6]);
            Rz=1e-1;
            Ts=1;
            Pz=diag([1,1]);
            Iz=eye(2);
            innovation=zeros(1,N);
            
            OCV = data.(cond).identified_para.OCV';
            fit=polyfit(SOC,OCV,9);
            diff_fit=polyder(fit);
            
            SOC = SOC(sp:end);

            Y = zeros(5, N);
            Y(:,1) = [3.6, 0, 1e-3, 1e-3, exp(-1)]; % initialize the model parameters
            Qy = diag([1e-2, 1e-4, 1e-6, 1e-6, 1e-6]);
            Ry = 1e-2;
            Py = eye(5);
            for i = 2:N % EKF algorithm operates
                % parameter identification
                f = diag([1, Y(5, i-1), 1, 1, 1]);
                g = [0, Y(4, i-1)*(1 - Y(5, i-1)), 0, 0, 0]';
                Yestimate = f*Y(:, i-1) + g*I(i-1);
                Uestimate = Yestimate(1) - Yestimate(2) - I(i)*Yestimate(3);
                e = Ut(i) - Uestimate;
                Fy = [1, 0, 0, 0, 0; 0, Yestimate(5), 0, I(i-1)*(1 - Yestimate(5)), Yestimate(2)-I(i-1)*Yestimate(4); ...
                    0, 0, 1, 0, 0; 0, 0, 0, 1, 0; 0, 0, 0, 0, 1];
                Hy = [1, -1,-I(i), 0, 0];
                Py1 = Fy*Py*Fy' + Qy;
                Ky = Py1*Hy'/(Hy*Py1*Hy' + Ry);
                Y(:, i) = Yestimate + Ky*e;
                Py = (eye(5) - Ky*Hy)*Py1;

                %SOC估算
                f=diag([Y(5,i),1]);
                g=[Y(4,i)*(1-Y(5,i)),-1/(Cn*3600)]';
                Zestimate=f*Z(:,i-1)+g*I(i-1); %状态量预测
                Uestimate=polyval(fit,Zestimate(2))-Zestimate(1)-I(i)*Y(3,i); %观测预测
                Fz=[Y(5,i), 0;0,1]; %状态转移矩阵
                Hz=[1,polyval(diff_fit,Zestimate(2))]; %观测矩阵
                Pz1=Fz*Pz*Fz'+Qz; %协方差阵一步更新
                Kz=Pz1*Hz'/(Hz*Pz1*Hz'+Rz); %卡尔曼增益
                e=Ut(i)-Uestimate; %计算新息
                innovation(i)=e; %储存新息
                Z(:,i)=Zestimate+Kz*e; %状态修正
                Pz=(Iz-Kz*Hz)*Pz1; %协方差阵修正
            end
            SOC_est_noKF_1 = Z(2,:)';
            absolute_error_noKF_1 = SOC_est_noKF_1 - SOC;

            result_all.(strcat('s_', num2str(start_point*100))).('m_4').result.(strcat('Temp_',temp)).(cond).soc_noKF = SOC_est_noKF_1;
            result_all.(strcat('s_', num2str(start_point*100))).('m_4').result.(strcat('Temp_',temp)).(cond).soc_real = SOC;
            result_all.(strcat('s_', num2str(start_point*100))).('m_4').result.(strcat('Temp_',temp)).(cond).err_noKF = absolute_error_noKF_1;
            result_all.(strcat('s_', num2str(start_point*100))).('m_4').result.(strcat('Temp_',temp)).(cond).cap_per = cap_rec_1;

            result_all.(strcat('s_', num2str(start_point*100))).('m_4').result.mae_noKF(temp_index, cond_index) = mae(absolute_error_noKF_1);
            result_all.(strcat('s_', num2str(start_point*100))).('m_4').result.rmse_noKF(temp_index, cond_index) = rms(absolute_error_noKF_1);
            mae_no_KF_all = [mae_no_KF_all; mae(absolute_error_noKF_1)];
            rmse_no_KF_all = [rmse_no_KF_all; rms(absolute_error_noKF_1)];

        end
    end
    result_all.(strcat('s_', num2str(start_point*100))).('m_4').result.mae_noKF_all = mae_no_KF_all;
    result_all.(strcat('s_', num2str(start_point*100))).('m_4').result.rmse_no_KF_all = rmse_no_KF_all;
end
save('.\resultsStorage\different_base_error_and_EKF_new.mat', "result_all")

%%

function [train_result, result]=func_base_error(soc_ini, start_point, method_flag, data, TEMP_ERROR_GROUP, TEMP_GROUP, CONDITION_GROUP)

%% training
% base modeling
SELECTED_HI_INDEX = [1,3,6];

switch method_flag
    case 1
        input_data = data.DST.different_window.w_300.input;
        output_data = data.DST.different_window.w_300.output;
        input_data = input_data(:, SELECTED_HI_INDEX);
    case 2
        input_data = data.DST.extracted_data(:, 1:2);
        input_data = [input_data, data.DST.identified_para.OCV'];
        output_data = data.DST.extracted_data(:, 3);
end

mu_in = mean(input_data);
sig_in = std(input_data);
x_train = (input_data-mu_in)./sig_in;
y_train = output_data;

t_s_base = tic;
base_model = fitrtree(x_train, y_train);
t_e_base = toc(t_s_base);
y_pred_base = predict(base_model,x_train);
absolute_error = y_pred_base - y_train;

train_result.base.result = [y_train, y_pred_base, absolute_error]; % [real, est, AE]
train_result.base.time = t_e_base;
train_result.base.model = base_model;

% error modeling
err_input = [];
err_output = [];

for temp_error_index = 1:numel(TEMP_ERROR_GROUP)
    temp_tmp_index = TEMP_ERROR_GROUP{temp_error_index};
    load(strcat('.\processedData\data_t_',temp_tmp_index,'_idx_1.mat'))

    switch method_flag
        case 1
            input_data = data.DST.different_window.w_300.input;
            output_data = data.DST.different_window.w_300.output;
            input_data = input_data(:, SELECTED_HI_INDEX);
        case 2
            input_data = data.DST.extracted_data(:, 1:2);
            input_data = [input_data, data.DST.identified_para.OCV'];
            output_data = data.DST.extracted_data(:, 3);
    end

    X = (input_data - mu_in)./sig_in;

    SOC_est_base_1 = predict(base_model, X);
    if strcmp(temp_tmp_index, 'N10')
        Temp1 = ones(numel(SOC_est_base_1),1)*(-10);
    else
        Temp1 = ones(numel(SOC_est_base_1),1)*(str2double(temp_tmp_index));
    end
    err_base = output_data - SOC_est_base_1;
    err_input = [err_input; Temp1, SOC_est_base_1];
    err_output = [err_output; err_base];
    train_result.error.result.(strcat('t_',temp_tmp_index)) = [Temp1, output_data, SOC_est_base_1, err_base]; % [temp, real, base_est, base_error]
end

t_s_error = tic;
err_model = fitrtree(err_input, err_output);
t_e_error = toc(t_s_error);
train_result.error.time = t_e_error;
train_result.error.model = err_model;

% add error estimation result on training data
for temp_error_index = 1:numel(TEMP_ERROR_GROUP)
    temp_tmp_index = TEMP_ERROR_GROUP{temp_error_index};
    err_est = predict(err_model, train_result.error.result.(strcat('t_',temp_tmp_index))(:, [1, 3]));
    train_result.error.result.(strcat('t_',temp_tmp_index))(:, 5) = err_est; % [..., error_est]
    train_result.error.result.(strcat('t_',temp_tmp_index))(:, 6) = err_est - train_result.error.result.(strcat('t_',temp_tmp_index))(:, 4); % [..., train absolute error]
end

%% Test
mae_no_KF_all = [];
rmse_no_KF_all = [];
for test_index = [1, 4, 7]
    temp_tmp_index = TEMP_GROUP{1,test_index};

    load(strcat('.\processedData\data_t_',temp_tmp_index,'_idx_1.mat'))
    for cond_index = 2:3  % remove DST
        clearvars SOC_est_noKF SOC_est_KF
        cond = CONDITION_GROUP{cond_index};
        output_data = data.(cond).extracted_data(:, 3);
        sp = find(output_data <= start_point, 1);
        output_data = output_data(sp:end, :);
        Qd_rec_1 = data.(cond).extracted_data(end,end)-data.(cond).extracted_data(1,end);
        Cn = Qd_rec_1 * (1+(4*rand-2)/100);
        N = numel(output_data);

        if strcmp(temp_tmp_index, 'N10')
            Temp1 = (-10);
        else
            Temp1 = (str2double(temp_tmp_index));
        end

        Ut = data.(cond).extracted_data(sp:end, 2);
        I = -data.(cond).extracted_data(sp:end, 1);
        
        % EKF-based modeling
        Y = zeros(5, N);
        Y(:,1) = [3.6, 0, 1e-3, 1e-3, exp(-1)]; % initialize the model parameters
        Qy = diag([1e-2, 1e-4, 1e-6, 1e-6, 1e-6]);
        Ry = 1e-2;
        Py = eye(5);

        P_kf = 1;

        Q_kf = 1e-6;
        R_kf = 1e-2;

        for i = 1:N
           if i == 1
               SOC_est_noKF(i, :) = soc_ini - I(i)/3600/Cn;
               SOC_est_KF = SOC_est_noKF;
               continue
           end
            f = diag([1, Y(5, i-1), 1, 1, 1]);
            g = [0, Y(4, i-1)*(1 - Y(5, i-1)), 0, 0, 0]';
            Yestimate = f*Y(:, i-1) + g*I(i-1);
            Uestimate = Yestimate(1) - Yestimate(2) - I(i)*Yestimate(3);
            e = Ut(i) - Uestimate;
            Fy = [1, 0, 0, 0, 0; 0, Yestimate(5), 0, I(i-1)*(1 - Yestimate(5)), Yestimate(2)-I(i-1)*Yestimate(4); ...
                0, 0, 1, 0, 0; 0, 0, 0, 1, 0; 0, 0, 0, 0, 1];
            Hy = [1, -1,-I(i), 0, 0];
            Py1 = Fy*Py*Fy' + Qy;
            Ky = Py1*Hy'/(Hy*Py1*Hy' + Ry);
            Y(:, i) = Yestimate + Ky*e;
            Py = (eye(5) - Ky*Hy)*Py1;

            switch method_flag
                case 1
                    if i < 300
                        SOC_est_noKF(i, :) = SOC_est_noKF(i-1, :) - I(i)/3600/Cn;
                        SOC_est_KF(i, :) = SOC_est_noKF(i, :);
                        continue
                    else
                        HI_max = max(Y(1, i - 300 + 1:i)); % maximum
                        HI_average = mean(Y(1, i - 300 + 1:i)); % average
                        p = polyfit(1:1:300, Y(1, i - 300 + 1:i), 1);
                        HI_intercept = p(2);
                        input_data = [HI_max, HI_average, HI_intercept];
                    end
                case 2
                    input_data = [Ut(i), I(i), Y(1, i)];
            end
            X1 = (input_data - mu_in)./sig_in;
            SOC_est_base_1 = predict(base_model, X1);
            errEst1 = predict(err_model, [Temp1, SOC_est_base_1]);
            SOC_est_noKF_1 = errEst1 + SOC_est_base_1;
            SOC_est_noKF(i, :) = SOC_est_noKF_1;
            
            socAh = SOC_est_KF(i-1)+(output_data(i)-output_data(i-1))/Cn/3600;
            P1 = P_kf+Q_kf;
            K = P1/(P1+R_kf);
            SOC_est_KF(i,:) = K*(SOC_est_noKF_1-socAh)+socAh;
            P_kf = (1-K)*P1;
        end
        

        absolute_error_KF_1 = SOC_est_KF - output_data;
        absolute_error_noKF_1 = SOC_est_noKF - output_data;

        result.(strcat('Temp_',temp_tmp_index)).(cond).err_KF = absolute_error_KF_1;
        result.(strcat('Temp_',temp_tmp_index)).(cond).soc_KF = SOC_est_KF;
        result.(strcat('Temp_',temp_tmp_index)).(cond).soc_noKF = SOC_est_noKF;
        result.(strcat('Temp_',temp_tmp_index)).(cond).soc_real = output_data;
        result.(strcat('Temp_',temp_tmp_index)).(cond).err_noKF = absolute_error_noKF_1;

        result.mae_KF(test_index, cond_index) = mae(absolute_error_KF_1);
        result.rmse_KF(test_index, cond_index) = rms(absolute_error_KF_1);
        result.mae_noKF(test_index, cond_index) = mae(absolute_error_noKF_1);
        result.rmse_noKF(test_index, cond_index) = rms(absolute_error_noKF_1);
        mae_no_KF_all = [mae_no_KF_all; mae(absolute_error_noKF_1)];
        rmse_no_KF_all = [rmse_no_KF_all; rms(absolute_error_noKF_1)];
    end
end
result.mae_noKF_all = mae_no_KF_all;
result.rmse_no_KF_all = rmse_no_KF_all;

end
