% Fine-tuning
% source model: 20degC DST 300windowsize

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
temp = '20';

fname = strcat('.\processedData\data_t_',temp,'_idx_1.mat');
load(fname)

input_data = data.(cond).different_window.(strcat('w_', num2str(windsize))).input;
output_data = data.(cond).different_window.(strcat('w_', num2str(windsize))).output;

input_data = input_data(:, SELECTED_HI_INDEX);
mu_in = mean(input_data);
sig_in = std(input_data);
x_train = (input_data-mu_in)./sig_in;
y_train = output_data;

layers = [
    sequenceInputLayer(size(x_train, 2))
    convolution1dLayer(8,8, Padding="same" )
    sigmoidLayer
    convolution1dLayer(128,128, Padding="same" )
    convolution1dLayer(256,256, Padding="same" )
    reluLayer
    convolution1dLayer(128,128, Padding="same" )
    reluLayer
    convolution1dLayer(64,64, Padding="same" )
    reluLayer
    convolution1dLayer(32,32, Padding="same" )
    sigmoidLayer
    fullyConnectedLayer(1)
    regressionLayer
    ];
opts = trainingOptions(...
    "adam",...
    "MaxEpochs",500,...
    "InitialLearnRate",1e-4,...
    "LearnRateSchedule","piecewise",...
    "LearnRateDropPeriod",125);
t_s_base = tic;
fine_tune_base_model = trainNetwork(x_train',y_train', layers,opts);
t_e_base = toc(t_s_base);
y_pred_base = predict(fine_tune_base_model,x_train');
y_pred_base = y_pred_base';
absolute_error = y_pred_base - y_train;

plot(y_pred_base-y_train,'*')

train_result.base.result = [y_train, y_pred_base, absolute_error]; % [real, est, AE]
train_result.base.time = t_e_base;
train_result.base.model = fine_tune_base_model;


%%
for temp_index = 1: 7
    if temp_index == 4
        continue
    end
    tempNeed = TEMP_GROUP{temp_index};

    load(strcat('.\processedData\data_t_',tempNeed,'_idx_1.mat'))
    input_data = data.(cond).different_window.(strcat('w_', num2str(windsize))).input;
    output_data = data.(cond).different_window.(strcat('w_', num2str(windsize))).output;

    X = (input_data(:, SELECTED_HI_INDEX) - mu_in)./sig_in;
    Y_train = output_data;
    layers = fine_tune_base_model.Layers;
    for i = 1:numel(layers)
        if  i == 2 || i == 4 || i ==5
            continue
        end
        finames = fieldnames(fine_tune_base_model.Layers(i,1));
        checkIdx = contains(finames,'Learn');
        j = find(checkIdx == 1);
        for k = 1:numel(j)
            layers(i).(finames{j(k)}) = 0;
        end
    end
    opts = trainingOptions(...
        "adam",...
        "MaxEpochs",500,...
        "InitialLearnRate",1e-4,...
        "LearnRateSchedule","piecewise",...
        "LearnRateDropPeriod",125);
    t_s_ft = tic;
    transferOtherMdl = trainNetwork(X', Y_train', layers, opts);
    t_e_ft = toc(t_s_ft);
    y_pred = predict(transferOtherMdl, X');
    y_pred = y_pred';

    train_result.ft.model_group{temp_index, 1} = transferOtherMdl;
    train_result.ft.time(temp_index, 1) = t_e_ft;
    train_result.ft.result.(strcat('t_',tempNeed)) = [Y_train, y_pred];
    train_result.ft.mae(temp_index, 1) = mae(y_pred-Y_train);
    train_result.ft.rms(temp_index, 1) = rms(y_pred-Y_train);
end

disp('all fine-tuning models trained!')

%% test

for tempIdx = 1:7
    tempNeed = TEMP_GROUP{1,tempIdx};
    if tempIdx == 4
        currentMdl = train_result.base.model;
    else
        currentMdl = train_result.ft.model_group{tempIdx, 1};
    end
    load(strcat('.\processedData\data_t_',tempNeed,'_idx_1.mat'))

    for cond_idx = 1:3
        current_condition = CONDITION_GROUP{cond_idx};
        input_data = data.(current_condition).different_window.(strcat('w_', num2str(windsize))).input;
        output_data = data.(current_condition).different_window.(strcat('w_', num2str(windsize))).output;
        X1 = (input_data(:, SELECTED_HI_INDEX) - mu_in)./sig_in;
        I1 = output_data;
        Qd_rec_1 = data.(current_condition).extracted_data(end,end)-data.(current_condition).extracted_data(1,end);
        soc_est_noKF_1 = predict(currentMdl, X1');
        soc_est_noKF_1 = soc_est_noKF_1';
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
save('.\resultsStorage\fine_tuning_result.mat',"result", "train_result")
disp('Test on transfer model complete!')

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


