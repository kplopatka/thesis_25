%% SVM Learning with Mag Separation

% Features extracted from "all_vectors.m"
% Data from BreaKHis Dataset

%{ 

SETUP OF FEATURE VECTOR:

~ size(feature_vector) = (7909,255)

~ 7909 total images, with 252 unique features, and 3 labels

~ 252 = 4 coeffs * 7 levels * 9 statistical parameters

~ (end - 3) = final feature point

~ Label 1: (end - 2) = magnification label:
            k = 1 --> 100x
            k = 2 --> 200x
            k = 3 --> 400x
            k = 4 -->  40x

~ Label 2: (end - 1) = patient ID label --> from 1 to 82

~ Label 3: (end)     = 0|1, where: 
            
                     -1 = benign
                      1 = malignant

%}

clear all; 

load("haar_vectors.mat");

% Separate by Mag Level
[M,N] = size(haar_vectors);

% Define number of patients - 82
num_patients = 82;

% Make placeholder feature_vector arrays
X100 = zeros(1,N); a = 1;
X200 = zeros(1,N); b = 1;
X400 = zeros(1,N); c = 1;
X40  = zeros(1,N); d = 1;

% This sorts the labeled vectors by magnification factor
for i = 1:M
    if haar_vectors(i,N-2) == 1
        X100(a,:) = haar_vectors(i,:);
        a = a + 1;
    end

    if haar_vectors(i,N-2) == 2
        X200(b,:) = haar_vectors(i,:);
        b = b + 1;
    end

    if haar_vectors(i,N-2) == 3
        X400(c,:) = haar_vectors(i,:);
        c = c + 1;
    end

    if haar_vectors(i,N-2) == 4
        X40(d,:) = haar_vectors(i,:);
        d = d + 1;
    end
end

[row_100,col_100] = size(X100);
[row_200,col_200] = size(X200);
[row_400,col_400] = size(X400);
[row_40,col_40] = size(X40);

% X100 --> Holds the feature vectors for all magnifications of 100x
% X200 --> Holds the feature vectors for all magnifications of 200x
% X400 --> Holds the feature vectors for all magnifications of 400x
%  X40 --> Holds the feature vectors for all magnifications of  40x

% 70% of the 82 Patients will be training data
num_train_patients = round(num_patients*0.7);

% 30% of the 82 Patients will be testing data
num_test_patients = num_patients - num_train_patients;


%% 100X SVM
tic;

% Partition the Data
rand_100 = randperm(num_patients);

% Separate Training and Testing Data
train_patients_labels_100 = rand_100(1:num_train_patients);
test_patients_labels_100 = rand_100(num_train_patients+1:end);

% Partition Data
train_patient_rows_100 = ismember(X100(:,end-1),train_patients_labels_100);
test_patient_rows_100  = ismember(X100(:,end-1),test_patients_labels_100);

train_patient_100 = X100(train_patient_rows_100,:);
test_patient_100 = X100(test_patient_rows_100,:);

% Preprocess for SVM
train_patient_features_100 = train_patient_100(:,1:end-3);
train_patient_labels_100 = train_patient_100(:,end);

test_patient_features_100 = test_patient_100(:,1:end-3);
test_patient_labels_100 = test_patient_100(:,end);

% Define Model
model_100 = fitcsvm(train_patient_features_100, train_patient_labels_100, 'KernelFunction', 'linear', ...
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

% Partition the Data
rand_200 = randperm(num_patients);

% Separate Training and Testing Data
train_patients_labels_200 = rand_200(1:num_train_patients);
test_patients_labels_200 = rand_200(num_train_patients+1:end);

% Partition Data
train_patient_rows_200 = ismember(X200(:,end-1),train_patients_labels_200);
test_patient_rows_200  = ismember(X200(:,end-1),test_patients_labels_200);

train_patient_200 = X200(train_patient_rows_200,:);
test_patient_200 = X200(test_patient_rows_200,:);

% Preprocess for SVM
train_patient_features_200 = train_patient_200(:,1:end-3);
train_patient_labels_200 = train_patient_200(:,end);

test_patient_features_200 = test_patient_200(:,1:end-3);
test_patient_labels_200 = test_patient_200(:,end);

% Define Model
model_200 = fitcsvm(train_patient_features_200, train_patient_labels_200, 'KernelFunction', 'linear', ...
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

% Partition the Data
rand_400 = randperm(num_patients);

% Separate Training and Testing Data
train_patients_labels_400 = rand_400(1:num_train_patients);
test_patients_labels_400 = rand_400(num_train_patients+1:end);

% Partition Data
train_patient_rows_400 = ismember(X400(:,end-1),train_patients_labels_400);
test_patient_rows_400  = ismember(X400(:,end-1),test_patients_labels_400);

train_patient_400 = X400(train_patient_rows_400,:);
test_patient_400 = X400(test_patient_rows_400,:);

% Preprocess for SVM
train_patient_features_400 = train_patient_400(:,1:end-3);
train_patient_labels_400 = train_patient_400(:,end);

test_patient_features_400 = test_patient_400(:,1:end-3);
test_patient_labels_400 = test_patient_400(:,end);

% Define Model
model_400 = fitcsvm(train_patient_features_400, train_patient_labels_400, 'KernelFunction', 'linear', ...
    'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', ...
    'expected-improvement-plus', 'ShowPlots', true));

% Test Model
result_400 = predict(model_400, test_patient_features_400);

% Get Total Count
[total_count400,~] = size(result_400);
%%
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

% Partition the Data
rand_40 = randperm(num_patients);

% Separate Training and Testing Data
train_patients_labels_40 = rand_40(1:num_train_patients);
test_patients_labels_40 = rand_40(num_test_patients+1:end);

% Partition Data
train_patient_rows_40 = ismember(X40(:,end-1),train_patients_labels_40);
test_patient_rows_40  = ismember(X40(:,end-1),test_patients_labels_40);

train_patient_40 = X40(train_patient_rows_40,:);
test_patient_40 = X40(test_patient_rows_40,:);

% Preprocess for SVM
train_patient_features_40 = train_patient_40(:,1:end-3);
train_patient_labels_40 = train_patient_40(:,end);

test_patient_features_40 = test_patient_40(:,1:end-3);
test_patient_labels_40 = test_patient_40(:,end);

% Define Model
model_40 = fitcsvm(train_patient_features_40, train_patient_labels_40, 'KernelFunction', 'linear', ...
    'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', ...
    'expected-improvement-plus', 'ShowPlots', true));

% Test Model
result_40 = predict(model_40, test_patient_features_40);

% Get Total Count
[total_count40,~] = size(result_40);
%%
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
accuracy = [accuracy_40,accuracy_100,accuracy_200,accuracy_400];
recall = [recall_40,recall_100,recall_200,recall_400];
false_alarm = [false_alarm_40,false_alarm_100,false_alarm_200,false_alarm_400];
precision = [precision_40,precision_100,precision_200,precision_400];






