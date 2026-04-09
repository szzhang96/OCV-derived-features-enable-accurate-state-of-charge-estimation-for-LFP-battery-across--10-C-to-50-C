% Unified model


clear
close all

SEED_INDEX = 1;

SELECTED_HI_INDEX = [1,3,6];
CONDITION_GROUP = {'DST', 'FUDS', 'US06'};
TEMP_GROUP = {'N10', '0', '10', '20', '30', '40', '50'};
WINDOW_GROUP = 100:100:500;

rng(SEED_INDEX)

cond = 'DST';
windsize = 300;

Inputs = [];
Outputs = [];
for tempIdx = 1:7
    temp = TEMP_GROUP{tempIdx};
    fname = strcat('.\processedData\data_t_',temp,'_idx_1.mat');
    load(fname)
    
    input_data = data.(cond).different_window.(strcat('w_', num2str(windsize))).input;
    output_data = data.(cond).different_window.(strcat('w_', num2str(windsize))).output;
    if tempIdx == 1
        input_t = -10;
    else
        input_t = str2double(temp);
    end
    Inputs = [Inputs; input_data, ones(numel(output_data),1)*input_t];
    Outputs = [Outputs; output_data];
end

Inputs = Inputs(:, [SELECTED_HI_INDEX, end]);
mu_in = mean(Inputs);
sig_in = std(Inputs);
x_train = (Inputs-mu_in)./sig_in;
y_train = Outputs;

t_s = tic;
unified_model = fitrtree(x_train,y_train);
t_e = toc(t_s);
y_pred_base = predict(unified_model,x_train);
absolute_error = y_pred_base - y_train;

plot(y_pred_base-y_train,'*')

train_result.result = [y_train, y_pred_base, absolute_error]; % [real, est, AE]
train_result.time = t_e;
train_result.model = unified_model;

%% test
for tempIdx = 1:7
    tempNeed = TEMP_GROUP{1,tempIdx};
    load(strcat('.\processedData\data_t_',tempNeed,'_idx_1.mat'))
    if tempIdx == 1
        input_t = -10;
    else
        input_t = str2double(tempNeed);
    end
    for cond_idx = 1:3
        current_condition = CONDITION_GROUP{cond_idx};
        input_data = data.(current_condition).different_window.(strcat('w_', num2str(windsize))).input;
        output_data = data.(current_condition).different_window.(strcat('w_', num2str(windsize))).output;
        X1 = ([input_data(:, SELECTED_HI_INDEX), ones(numel(output_data),1)*input_t] - mu_in)./sig_in;
        I1 = output_data;
        Qd_rec_1 = data.(current_condition).extracted_data(end,end)-data.(current_condition).extracted_data(1,end);
        soc_est_noKF_1 = predict(unified_model, X1);
        [resultWithKF_1,cap_rec_1] = getFinalSocEst(soc_est_noKF_1,I1, Qd_rec_1);

        absolute_error_KF_1 = resultWithKF_1 - I1;
        absolute_error_noKF_1 = soc_est_noKF_1 - I1;
        result.(strcat('Temp_',tempNeed)).(current_condition).err_KF = absolute_error_KF_1;
        result.(strcat('Temp_',tempNeed)).(current_condition).soc_KF = resultWithKF_1;
        result.(strcat('Temp_',tempNeed)).(current_condition).soc_noKF = soc_est_noKF_1;
        result.(strcat('Temp_',tempNeed)).(current_condition).soc_real = I1;
        result.(strcat('Temp_',tempNeed)).(current_condition).err_noKF = absolute_error_noKF_1;
        result.(strcat('Temp_',tempNeed)).(current_condition).cap_per = cap_rec_1;

        result.mae_KF(tempIdx, cond_idx) = mae(absolute_error_KF_1(:, 2));
        result.rmse_KF(tempIdx, cond_idx) = rms(absolute_error_KF_1(:, 2));
        result.mae_noKF(tempIdx, cond_idx) = mae(absolute_error_noKF_1);
        result.rmse_noKF(tempIdx, cond_idx) = rms(absolute_error_noKF_1);
    end
end
save('.\resultsStorage\unified_result_0408.mat',"result", "train_result")
disp('Test on unified model complete!')

function [resultWithKF_1,cap_rec] = getFinalSocEst(soc_est_noKF_1,I1, Qd_rec_1)
nnum = 1;
P = 1;

Q = 1e-6;
R = 1e-2;
cap_rec = [];
for coCap = [1 1+(4*rand-2)/100]
    for k = 1:numel(soc_est_noKF_1)
        resultWithKF_1(nnum, k) = soc_est_noKF_1(k);
        if k < 2
            resultWithKF_1(nnum, k) = soc_est_noKF_1(k);
            continue
        end
        socAh = resultWithKF_1(nnum, k-1)+(I1(k)-I1(k-1))/Qd_rec_1/coCap/3600;
        P1 = P+Q;
        K = P1/(P1+R);
        resultWithKF_1(nnum, k) = K*(soc_est_noKF_1(k)-socAh)+socAh;
        P = (1-K)*P1;
    end
    cap_rec(nnum) = Qd_rec_1/coCap;
    nnum = nnum+1;
end
resultWithKF_1 = resultWithKF_1';
end


