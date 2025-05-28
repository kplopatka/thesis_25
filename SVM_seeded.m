%% SVM ~ Standard Testing with Mag Separation

% Features extracted from "all_vectors.m"
% Data from BreaKHis Dataset

%{ 

SETUP OF FEATURE VECTOR:

~ size(feature_vector) = (7909,759)

~ 7909 total images, with 756 unique features, and 3 labels

~ 756 = 3 color layers * 4 coeffs * 7 levels * 9 statistical parameters

~ (end - 3) = final feature point

~ Label 1: (end - 2) = magnification label:
            k = 1 --> 100x
            k = 2 --> 200x
            k = 3 --> 400x
            k = 4 -->  40x

~ Label 2: (end - 1) = patient ID label --> from 1 to 82

~ Label 3: (end)     = -1|1, where: 
            
                     -1 = benign
                      1 = malignant

%}

clear; 

load("rgb_haar_vectors.mat");
for x = 1:5
    % Set a seed for consistent testing
    seed = (1:5);
    rng(seed(x));
    
    % Separate by Mag Level
    [M,N] = size(rgb_haar_vectors);
    
    % Define number of patients - 82
    num_patients = 82;
    
    X100 = rgb_haar_vectors(rgb_haar_vectors(:,end-2) == 1,:);
    X200 = rgb_haar_vectors(rgb_haar_vectors(:,end-2) == 2,:);
    X400 = rgb_haar_vectors(rgb_haar_vectors(:,end-2) == 3,:);
    X40 = rgb_haar_vectors(rgb_haar_vectors(:,end-2) == 4,:);
    
    benign = (1:24);
    malignant = (25:82);
    
    % X100 --> Holds the feature vectors for all magnifications of 100x
    % X200 --> Holds the feature vectors for all magnifications of 200x
    % X400 --> Holds the feature vectors for all magnifications of 400x
    %  X40 --> Holds the feature vectors for all magnifications of  40x
    
    %% Partition Method 1
    %{
    % 70% of the 82 Patients will be training data
    num_train_patients = round(num_patients*0.7);
    
    % 30% of the 82 Patients will be testing data
    num_test_patients = num_patients - num_train_patients;
    
    % Partition the Data
    random_patient_labels = randperm(num_patients);
    
    % Separate Training and Testing Data
    train_patient_IDs = random_patient_labels(1:num_train_patients);
    test_patient_IDs = random_patient_labels(num_train_patients+1:end);
    %}
    
    %% Partition Method 2
    randomized_malignant = malignant(randperm(length(malignant)));
    randomized_benign = benign(randperm(length(benign)));
    
    partition_ratio = 0.7;
    mal_len = length(randomized_malignant);
    ben_len = length(randomized_benign);
    
    ans1 = round(partition_ratio*mal_len);
    ans2 = round(partition_ratio*ben_len);
    
    train_patient_IDs = horzcat(randomized_malignant(1:ans1),randomized_benign(1:ans2));
    test_patient_IDs = horzcat(randomized_malignant(ans1+1:end),randomized_benign(ans2+1:end));
    
    %% 100X SVM
    tic;
    
    % Partition Data
    train_patient_rows_100 = ismember(X100(:,end-1),train_patient_IDs);
    test_patient_rows_100  = ismember(X100(:,end-1),test_patient_IDs);
    
    train_patient_100 = X100(train_patient_rows_100,:);
    test_patient_100 = X100(test_patient_rows_100,:);
    
    % Preprocess for SVM
    train_patient_features_100 = train_patient_100(:,1:end-3);
    train_patient_labels_100 = train_patient_100(:,end);
    
    test_patient_features_100 = test_patient_100(:,1:end-3);
    test_patient_labels_100 = test_patient_100(:,end);
    
    % Define Model
    model_100 = fitcsvm(train_patient_features_100, train_patient_labels_100, 'KernelFunction', 'rbf', ...
        'OptimizeHyperparameters', 'auto', ...
        'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', ...
        'expected-improvement-plus', 'ShowPlots', true));
    
    % Test Model
    result_100 = predict(model_100, test_patient_features_100);
    
    % Get Total Count
    [total_count100,~] = size(result_100);
    
    TP100 = 0;
    TN100 = 0;
    FP100 = 0;
    FN100 = 0;
    
    % Confusion Matrix
    for i = 1:total_count100
        if ((result_100(i,1) == 1) && (test_patient_labels_100(i,1) == 1))
            TP100 = TP100 + 1;
        elseif ((result_100(i,1) == -1) && (test_patient_labels_100(i,1) == 1))
            FN100 = FN100 + 1;
        elseif ((result_100(i,1) == 1) && (test_patient_labels_100(i,1) == -1))
            FP100 = FP100 + 1;
        elseif ((result_100(i,1) == -1) && (test_patient_labels_100(i,1) == -1))
            TN100 = TN100 + 1;
        end
    end
    
    % Results
    accuracy_100 = (TP100+TN100)/(TP100+TN100+FP100+FN100);
    recall_100 = (TP100)/(TP100+FN100);
    false_alarm_100 = FP100/(TP100+FP100);
    precision_100 = TP100/(TP100+FP100);
    
    time_100 = toc;
    
    %% 200 SVM
    tic;
    
    % Partition Data
    train_patient_rows_200 = ismember(X200(:,end-1),train_patient_IDs);
    test_patient_rows_200  = ismember(X200(:,end-1),test_patient_IDs);
    
    train_patient_200 = X200(train_patient_rows_200,:);
    test_patient_200 = X200(test_patient_rows_200,:);
    
    % Preprocess for SVM
    train_patient_features_200 = train_patient_200(:,1:end-3);
    train_patient_labels_200 = train_patient_200(:,end);
    
    test_patient_features_200 = test_patient_200(:,1:end-3);
    test_patient_labels_200 = test_patient_200(:,end);
    
    % Define Model
    model_200 = fitcsvm(train_patient_features_200, train_patient_labels_200, 'KernelFunction', 'rbf', ...
        'OptimizeHyperparameters', 'auto', ...
        'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', ...
        'expected-improvement-plus', 'ShowPlots', true));
    
    % Test Model
    result_200 = predict(model_200, test_patient_features_200);
    
    % Get Total Count
    [total_count200,~] = size(result_200);
    
    TP200 = 0;
    TN200 = 0;
    FP200 = 0;
    FN200 = 0;
    
    % Confusion Matrix
    for i = 1:total_count200
        if ((result_200(i,1) == 1) && (test_patient_labels_200(i,1) == 1))
            TP200 = TP200 + 1;
        elseif ((result_200(i,1) == -1) && (test_patient_labels_200(i,1) == 1))
            FN200 = FN200 + 1;
        elseif ((result_200(i,1) == 1) && (test_patient_labels_200(i,1) == -1))
            FP200 = FP200 + 1;
        elseif ((result_200(i,1) == -1) && (test_patient_labels_200(i,1) == -1))
            TN200 = TN200 + 1;
        end
    end
    
    % Results
    accuracy_200 = (TP200+TN200)/(TP200+TN200+FP200+FN200);
    recall_200 = (TP200)/(TP200+FN200);
    false_alarm_200 = FP200/(TP200+FP200);
    precision_200 = TP200/(TP200+FP200);
    
    time_200 = toc;
    
    %% 400 SVM
    tic;
    
    % Partition Data
    train_patient_rows_400 = ismember(X400(:,end-1),train_patient_IDs);
    test_patient_rows_400  = ismember(X400(:,end-1),test_patient_IDs);
    
    train_patient_400 = X400(train_patient_rows_400,:);
    test_patient_400 = X400(test_patient_rows_400,:);
    
    % Preprocess for SVM
    train_patient_features_400 = train_patient_400(:,1:end-3);
    train_patient_labels_400 = train_patient_400(:,end);
    
    test_patient_features_400 = test_patient_400(:,1:end-3);
    test_patient_labels_400 = test_patient_400(:,end);
    
    % Define Model
    model_400 = fitcsvm(train_patient_features_400, train_patient_labels_400, 'KernelFunction', 'rbf', ...
        'OptimizeHyperparameters', 'auto', ...
        'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', ...
        'expected-improvement-plus', 'ShowPlots', true));
    
    % Test Model
    result_400 = predict(model_400, test_patient_features_400);
    
    % Get Total Count
    [total_count400,~] = size(result_400);
    
    TP400 = 0;
    TN400 = 0;
    FP400 = 0;
    FN400 = 0;
    
    % Confusion Matrix
    for i = 1:total_count400
        if ((result_400(i,1) == 1) && (test_patient_labels_400(i,1) == 1))
            TP400 = TP400 + 1;
        elseif ((result_400(i,1) == -1) && (test_patient_labels_400(i,1) == 1))
            FN400 = FN400 + 1;
        elseif ((result_400(i,1) == 1) && (test_patient_labels_400(i,1) == -1))
            FP400 = FP400 + 1;
        elseif ((result_400(i,1) == -1) && (test_patient_labels_400(i,1) == -1))
            TN400 = TN400 + 1;
        end
    end
    
    % Results
    accuracy_400 = (TP400+TN400)/(TP400+TN400+FP400+FN400);
    recall_400 = (TP400)/(TP400+FN400);
    false_alarm_400 = FP400/(TP400+FP400);
    precision_400 = TP400/(TP400+FP400);
    
    time_400 = toc;
    
    %% 40X SVM
    tic;
    
    % Partition Data
    train_patient_rows_40 = ismember(X40(:,end-1),train_patient_IDs);
    test_patient_rows_40  = ismember(X40(:,end-1),test_patient_IDs);
    
    train_patient_40 = X40(train_patient_rows_40,:);
    test_patient_40 = X40(test_patient_rows_40,:);
    
    % Preprocess for SVM
    train_patient_features_40 = train_patient_40(:,1:end-3);
    train_patient_labels_40 = train_patient_40(:,end);
    
    test_patient_features_40 = test_patient_40(:,1:end-3);
    test_patient_labels_40 = test_patient_40(:,end);
    
    % Define Model
    model_40 = fitcsvm(train_patient_features_40, train_patient_labels_40, 'KernelFunction', 'rbf', ...
        'OptimizeHyperparameters', 'auto', ...
        'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', ...
        'expected-improvement-plus', 'ShowPlots', true));
    
    % Test Model
    result_40 = predict(model_40, test_patient_features_40);
    
    % Get Total Count
    [total_count40,~] = size(result_40);
    
    TP40 = 0;
    TN40 = 0;
    FP40 = 0;
    FN40 = 0;
    
    % Confusion Matrix
    for i = 1:total_count40
        if ((result_40(i,1) == 1) && (test_patient_labels_40(i,1) == 1))
            TP40 = TP40 + 1;
        elseif ((result_40(i,1) == -1) && (test_patient_labels_40(i,1) == 1))
            FN40 = FN40 + 1;
        elseif ((result_40(i,1) == 1) && (test_patient_labels_40(i,1) == -1))
            FP40 = FP40 + 1;
        elseif ((result_40(i,1) == -1) && (test_patient_labels_40(i,1) == -1))
            TN40 = TN40 + 1;
        end
    end
    
    % Results
    accuracy_40 = (TP40+TN40)/(TP40+TN40+FP40+FN40);
    recall_40 = (TP40)/(TP40+FN40);
    false_alarm_40 = FP40/(TP40+FP40);
    precision_40 = TP40/(TP40+FP40);
    
    time_40 = toc;
    
    %% Display Results
    accuracy = [accuracy_40,accuracy_100,accuracy_200,accuracy_400]
    recall = [recall_40,recall_100,recall_200,recall_400]
    false_alarm = [false_alarm_40,false_alarm_100,false_alarm_200,false_alarm_400]
    precision = [precision_40,precision_100,precision_200,precision_400]
    
    a_time = (time_40+time_100+time_200+time_400)/60;
    minutes = floor(a_time)
    seconds = 60*(a_time - minutes)

end



