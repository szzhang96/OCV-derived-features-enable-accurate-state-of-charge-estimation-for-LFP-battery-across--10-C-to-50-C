% For data preprocess
% 
% data split (DST, US06, FUDS)
% parameter identification (different temperature)
% feature extraction (different windowsize, different temperature)

clear
close all
warning off
mkdir processedData

% get the OCV-SOC under different tempertures, different windows size
tempNeedGroup = {'N10', '0', '10', '20', '25', '30', '40', '50'};
root_folder_path = '.\unzippedOriginalData\';
for tempIdx = 1:8
    tempNeed1 = tempNeedGroup{1,tempIdx};
    disp(datetime('now'))
    disp(['now solving: ', tempNeed1])
    fieldname_t = strcat('t_', tempNeed1);
    folderList = dir(root_folder_path);
    folderList(1:2,:) = [];
    tmpName = folderList.name;
    folderIdx = [];
    for i = 1:numel(folderList)
        if strfind(folderList(i,1).name,strcat('-',tempNeed1))
            tempNeed1FolderIdx = i;
        end
    end
    folderListSub1 = dir(strcat(root_folder_path,folderList(tempNeed1FolderIdx,1).name,'\'));
    folderListSub1(1:2,:) = [];
    fileList1 = dir(strcat(root_folder_path,folderList(tempNeed1FolderIdx,1).name,'\',folderListSub1(1,1).name,'\'));
    fileList1(1:2,:) = [];

    for i = 1:numel(fileList1)
        if strcmp(fileList1(i,1).name(1), '~')
            break
        end
        fieldname_i = strcat('idx_', num2str(i));
        fnameTmp = strcat(root_folder_path,folderList(tempNeed1FolderIdx,1).name,'\',folderListSub1(1,1).name,'\',fileList1(i,1).name);
        sheetTmp = sheetnames(fnameTmp);
        readSheetName = sheetTmp(3);
        dataTmp = readtable(fnameTmp,'Sheet',readSheetName,'VariableNamingRule','preserve');

        stepIdxData = dataTmp.("Step_Index");

        % find all the indexes needed in the data processing
        % - DST: step index 8
        % - US06: step index 16
        % - FUDS: 24

        DST_idx = find(stepIdxData == 8);
        US06_idx = find(stepIdxData == 16);
        FUDS_idx = find(stepIdxData == 24);

        data_DST_all = dataTmp(DST_idx,:);
        data_US06_all= dataTmp(US06_idx,:);
        data_FUDS_all= dataTmp(FUDS_idx,:);

        refSOC_DST = data_DST_all.("Discharge_Capacity(Ah)");
        refSOC_DST = refSOC_DST-refSOC_DST(1);
        refSOC_DST = refSOC_DST./refSOC_DST(end);
        refSOC_DST = flip(refSOC_DST);

        refSOC_US06 = data_US06_all.("Discharge_Capacity(Ah)");
        refSOC_US06 = refSOC_US06-refSOC_US06(1);
        refSOC_US06 = refSOC_US06./refSOC_US06(end);
        refSOC_US06 = flip(refSOC_US06);

        refSOC_FUDS = data_FUDS_all.("Discharge_Capacity(Ah)");
        refSOC_FUDS = refSOC_FUDS-refSOC_FUDS(1);
        refSOC_FUDS = refSOC_FUDS./refSOC_FUDS(end);
        refSOC_FUDS = flip(refSOC_FUDS);

        DATA_DST = [data_DST_all.("Current(A)"), data_DST_all.("Voltage(V)"), refSOC_DST, data_DST_all.("Discharge_Capacity(Ah)")];
        DATA_US06 = [data_US06_all.("Current(A)"), data_US06_all.("Voltage(V)"), refSOC_US06, data_US06_all.("Discharge_Capacity(Ah)")];
        DATA_FUDS = [data_FUDS_all.("Current(A)"), data_FUDS_all.("Voltage(V)"), refSOC_FUDS, data_FUDS_all.("Discharge_Capacity(Ah)")];
        [Y_DST, raw_identificationResults_DST, SOC_real_DST] = para_identify(DATA_DST);
        data.DST.original_data = data_DST_all;
        data.DST.extracted_data = DATA_DST;
        data.DST.identified_para = raw_identificationResults_DST;

        [Y_FUDS, raw_identificationResults_FUDS, SOC_real_FUDS] = para_identify(DATA_FUDS);
        data.FUDS.original_data = data_FUDS_all;
        data.FUDS.extracted_data = DATA_FUDS;
        data.FUDS.identified_para = raw_identificationResults_FUDS;

        [Y_US06, raw_identificationResults_US06, SOC_real_US06] = para_identify(DATA_US06);
        data.US06.original_data = data_US06_all;
        data.US06.extracted_data = DATA_US06;
        data.US06.identified_para = raw_identificationResults_US06;
        for windowSize = 100:100:900
            fieldname_w = strcat('w_', num2str(windowSize));
            fprintf(strcat(fieldname_w, '\t'))
            if windowSize == 900
                fprintf('\n')
            end
            [Inputs_DST, Outputs_DST, correlation] = dataProcess(Y_DST, windowSize, SOC_real_DST);
            data.DST.different_window.(fieldname_w).input = Inputs_DST;
            data.DST.different_window.(fieldname_w).output = Outputs_DST;
            data.DST.different_window.(fieldname_w).corr = correlation;

            [Inputs_FUDS, Outputs_FUDS, correlation] = dataProcess(Y_FUDS, windowSize, SOC_real_FUDS);
            data.FUDS.different_window.(fieldname_w).input = Inputs_FUDS;
            data.FUDS.different_window.(fieldname_w).output = Outputs_FUDS;
            data.FUDS.different_window.(fieldname_w).corr = correlation;

            [Inputs_US06, Outputs_US06, correlation] = dataProcess(Y_US06, windowSize, SOC_real_US06);
            data.US06.different_window.(fieldname_w).input = Inputs_US06;
            data.US06.different_window.(fieldname_w).output = Outputs_US06;
            data.US06.different_window.(fieldname_w).corr = correlation;

        end
        save(strcat('.\processedData\data_',fieldname_t,'_',fieldname_i,'.mat'), "data");
    end
end
disp('original data has been splited!')

function [Y, raw_identificationResults, SOC_real] = para_identify(data)
Ut = data(:, 2);
I = -data(:, 1);
SOC_real = data(:, 3); % Ampere hour method with corrected initial SOC, precise capacity and current
N = size(data, 1);

% EKF-based modeling
Y = zeros(5, N);
Y(:,1) = [3.6, 0, 1e-3, 1e-3, exp(-1)]; % initialize the model parameters
Qy = diag([1e-2, 1e-4, 1e-6, 1e-6, 1e-6]);
Ry = 1e-2;
Py = eye(5);
for i = 2:N % EKF algorithm operates
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
end

raw_identificationResults.OCV = Y(1,:);
raw_identificationResults.Up = Y(2,:);
raw_identificationResults.Ro = Y(3,:);
raw_identificationResults.Rp = Y(4,:);
raw_identificationResults.exp_tao = Y(5,:);
end

function [InputData, OutputData, correlation] = dataProcess(Y, windowSize, SOC_real)

convergence_time = 100;
SOC_real = SOC_real(convergence_time + 1 : end); % the SOC_real before convergence is removed
Y = Y(:, convergence_time + 1 : end); % the identified model parameters before convergence are removed
moving_window = windowSize; 

OutputData = [];
InputData = [];
for i = 1:size(Y, 2)
    if i >= moving_window
        Inputs_potential = []; 
        for j = 1:5
            HI_max = max(Y(j, i - moving_window + 1:i)); % maximum
            HI_min = min(Y(j, i - moving_window + 1:i)); % minimum
            HI_average = mean(Y(j, i - moving_window + 1:i)); % average
            HI_var = var(Y(j, i - moving_window + 1:i)); % variance
            p = polyfit(1:1:moving_window, Y(j, i - moving_window + 1:i), 1);
            HI_slope = p(1);
            HI_intercept = p(2);
            HI_cov = 100*std(Y(j, i - moving_window + 1:i))/HI_average; % OCV coefficient of variation
            HI_skew = skewness(Y(j, i - moving_window + 1:i)); % OCV skewness
            HI_kur = kurtosis(Y(j, i - moving_window + 1:i)); % OCV kurtosis
            Inputs_potential = [Inputs_potential, HI_max, HI_min, HI_average, ...
                HI_var, HI_slope, HI_intercept, HI_cov, HI_skew, HI_kur];
        end
        InputData = [InputData; Inputs_potential];
        OutputData = [OutputData; SOC_real(i)];
    end
end

% Correlation analysis
correlation = [];
for i = 1:size(InputData, 2)
    correlation = [correlation, corr(InputData(:, i), OutputData)];
end
end

