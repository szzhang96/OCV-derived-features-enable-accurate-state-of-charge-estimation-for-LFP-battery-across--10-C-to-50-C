% Base-error joint modeling

clear
clc
warning off

SEED_INDEX = 1;
TEMP_GROUP = {'N10', '0', '10', '20', '30', '40', '50'};
CONDITION_GROUP = {'DST', 'FUDS', 'US06'};
WINDOW_GROUP = 100:100:500;
SELECTED_HI_INDEX = [1,3,6];

rng(SEED_INDEX)


cnt = 1;
for condition_index = 1: 3
    cond = CONDITION_GROUP{condition_index};

    for temperature_index = 1:7
        temp = TEMP_GROUP{temperature_index};

        for window_index = 1:5
            disp(cnt)
            cnt = cnt+1;
            windsize = WINDOW_GROUP(window_index);

            TEMP_ERROR_GROUP = TEMP_GROUP;

            save_fname = strcat('.\resultsStorage\base_error_results_', ...
                cond,'_t_',temp,'_w_',num2str(windsize),'.mat');

            fname = strcat('.\processedData\data_t_',temp,'_idx_1.mat');
            load(fname)

            %% base modeling
            input_data = data.(cond).different_window.(strcat('w_', num2str(windsize))).input;
            output_data = data.(cond).different_window.(strcat('w_', num2str(windsize))).output;

            input_data = input_data(:, SELECTED_HI_INDEX);
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

            %% error modeling
            err_input = [];
            err_output = [];

            plt_err = [];

            for temp_error_index = 1:numel(TEMP_ERROR_GROUP)
                temp_tmp_index = TEMP_ERROR_GROUP{temp_error_index};
                load(strcat('.\processedData\data_t_',temp_tmp_index,'_idx_1.mat'))
                input_data = data.(cond).different_window.(strcat('w_', num2str(windsize))).input;
                output_data = data.(cond).different_window.(strcat('w_', num2str(windsize))).output;

                X = (input_data(:, SELECTED_HI_INDEX) - mu_in)./sig_in;

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
                train_result.error.result.(strcat('t_',temp_tmp_index))(:, 6) = err_est - train_result.error.result.(strcat('t_',temp_tmp_index))(:, 4); % [..., error absolute error]
            end

            % disp('error model trained!')

            %% test
            mkdir resultsStorage
            for test_index = 1:7
                temp_tmp_index = TEMP_GROUP{1,test_index};

                load(strcat('.\processedData\data_t_',temp_tmp_index,'_idx_1.mat'))
                input_data = data.DST.different_window.(strcat('w_', num2str(windsize))).input;
                output_data = data.DST.different_window.(strcat('w_', num2str(windsize))).output;
                X1 = (input_data(:,SELECTED_HI_INDEX) - mu_in)./sig_in;
                I1 = output_data;
                Qd_rec_1 = data.DST.extracted_data(end,end)-data.DST.extracted_data(1,end);
                SOC_est_base_1 = predict(base_model, X1);

                input_data = data.FUDS.different_window.(strcat('w_', num2str(windsize))).input;
                output_data = data.FUDS.different_window.(strcat('w_', num2str(windsize))).output;
                X2 = (input_data(:,SELECTED_HI_INDEX) - mu_in)./sig_in;
                I2 = output_data;
                Qd_rec_2 = data.FUDS.extracted_data(end,end)-data.FUDS.extracted_data(1,end);
                SOC_est_base_2 = predict(base_model,X2);

                input_data = data.US06.different_window.(strcat('w_', num2str(windsize))).input;
                output_data = data.US06.different_window.(strcat('w_', num2str(windsize))).output;
                X3 = (input_data(:,SELECTED_HI_INDEX) - mu_in)./sig_in;
                I3 = output_data;
                Qd_rec_3 = data.US06.extracted_data(end,end)-data.US06.extracted_data(1,end);
                SOC_est_base_3 = predict(base_model,X3);

                if strcmp(temp_tmp_index, 'N10')
                    Temp1 = ones(numel(SOC_est_base_1),1)*(-10);
                    Temp2 = ones(numel(SOC_est_base_2),1)*(-10);
                    Temp3 = ones(numel(SOC_est_base_3),1)*(-10);
                else
                    Temp1 = ones(numel(SOC_est_base_1),1)*(str2double(temp_tmp_index));
                    Temp2 = ones(numel(SOC_est_base_2),1)*(str2double(temp_tmp_index));
                    Temp3 = ones(numel(SOC_est_base_3),1)*(str2double(temp_tmp_index));
                end

                errInput1 = [Temp1, SOC_est_base_1];
                errEst1 = predict(err_model, errInput1);
                SOC_est_noKF_1 = errEst1 + SOC_est_base_1;

                errInput2 = [Temp2, SOC_est_base_2];
                errEst2 = predict(err_model, errInput2);
                SOC_est_noKF_2 = errEst2 + SOC_est_base_2;

                errInput3 = [Temp3, SOC_est_base_3];
                errEst3 = predict(err_model, errInput3);
                SOC_est_noKF_3 = errEst3 + SOC_est_base_3;

                [resultWithKF_1, cap_rec_1] = getFinalSocEst(SOC_est_noKF_1, I1, Qd_rec_1);
                [resultWithKF_2, cap_rec_2] = getFinalSocEst(SOC_est_noKF_2, I2, Qd_rec_2);
                [resultWithKF_3, cap_rec_3] = getFinalSocEst(SOC_est_noKF_3, I3, Qd_rec_3);

                absolute_error_KF_1 = resultWithKF_1 - I1;
                absolute_error_noKF_1 = SOC_est_noKF_1 - I1;

                absolute_error_KF_2 = resultWithKF_2 - I2;
                absolute_error_noKF_2 = SOC_est_noKF_2 - I2;

                absolute_error_KF_3 = resultWithKF_3 - I3;
                absolute_error_noKF_3 = SOC_est_noKF_3 - I3;

                result.(strcat('Temp_',temp_tmp_index)).DST.err_KF = absolute_error_KF_1;
                result.(strcat('Temp_',temp_tmp_index)).DST.soc_KF = resultWithKF_1;
                result.(strcat('Temp_',temp_tmp_index)).DST.soc_noKF = SOC_est_noKF_1;
                result.(strcat('Temp_',temp_tmp_index)).DST.soc_real = I1;
                result.(strcat('Temp_',temp_tmp_index)).DST.err_noKF = absolute_error_noKF_1;
                result.(strcat('Temp_',temp_tmp_index)).DST.cap_per = cap_rec_1;

                result.(strcat('Temp_',temp_tmp_index)).FUDS.err_KF = absolute_error_KF_2;
                result.(strcat('Temp_',temp_tmp_index)).FUDS.soc_KF = resultWithKF_2;
                result.(strcat('Temp_',temp_tmp_index)).FUDS.soc_noKF = SOC_est_noKF_2;
                result.(strcat('Temp_',temp_tmp_index)).FUDS.soc_real = I2;
                result.(strcat('Temp_',temp_tmp_index)).FUDS.err_noKF = absolute_error_noKF_2;
                result.(strcat('Temp_',temp_tmp_index)).FUDS.cap_per = cap_rec_2;

                result.(strcat('Temp_',temp_tmp_index)).US06.err_KF = absolute_error_KF_3;
                result.(strcat('Temp_',temp_tmp_index)).US06.soc_KF = resultWithKF_3;
                result.(strcat('Temp_',temp_tmp_index)).US06.soc_noKF = SOC_est_noKF_3;
                result.(strcat('Temp_',temp_tmp_index)).US06.soc_real = I3;
                result.(strcat('Temp_',temp_tmp_index)).US06.err_noKF = absolute_error_noKF_3;
                result.(strcat('Temp_',temp_tmp_index)).US06.cap_per = cap_rec_3;

                result.mae_KF(test_index, :) = [mae(absolute_error_KF_1(:, 2)), mae(absolute_error_KF_2(:, 2)), mae(absolute_error_KF_3(:, 2))];
                result.rmse_KF(test_index, :) = [rms(absolute_error_KF_1(:, 2)), rms(absolute_error_KF_2(:, 2)), rms(absolute_error_KF_3(:, 2))];
                result.mae_noKF(test_index, :) = [mae(absolute_error_noKF_1), mae(absolute_error_noKF_2), mae(absolute_error_noKF_3)];
                result.rmse_noKF(test_index, :) = [rms(absolute_error_noKF_1), rms(absolute_error_noKF_2), rms(absolute_error_noKF_3)];
            end
            save(save_fname, "train_result", "result")

        end
    end
end
disp('Test complete!')

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
