%% SVM Learning with Mag Separation

% Features extracted from "all_vectors.m"
% Data from BreaKHis Dataset

%{ 

Important Note Regarding Labeled Vector Set:

(end - 3) = final feature point

(end - 2) = magnification label:
            k = 1 --> 100x
            k = 2 --> 200x
            k = 3 --> 400x
            k = 4 -->  40x

(end - 1) = patient ID label --> from 1 to 82

(end)     = 0|1, where 
            
            0 = benign
            1 = malignant

%}

clear all; 

load("labeled_vectors.mat");

% Separate by Mag Level
[M,N] = size(labeled_vectors);

% Define number of patients - 82
num_patients = 82;

% Make placeholder feature_vector arrays
X100 = zeros(1,N); a = 1;
X200 = zeros(1,N); b = 1;
X400 = zeros(1,N); c = 1;
X40  = zeros(1,N); d = 1;

% This sorts the labeled vectors by magnification factor
for i = 1:M
    if labeled_vectors(i,N-2) == 1
        X100(a,:) = labeled_vectors(i,:);
        a = a + 1;
    end

    if labeled_vectors(i,N-2) == 2
        X200(b,:) = labeled_vectors(i,:);
        b = b + 1;
    end

    if labeled_vectors(i,N-2) == 3
        X400(c,:) = labeled_vectors(i,:);
        c = c + 1;
    end

    if labeled_vectors(i,N-2) == 4
        X40(d,:) = labeled_vectors(i,:);
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
rand = randperm(num_patients);

% Isolate Feature and Labels
features_X100 = X100(:,1:end-3);
class_labels = X100(:,end);

% Separate Training and Testing Data
train_patients_labels = rand(1:num_train_patients);
test_patients_labels = rand(num_train_patients+1:end);

% Partition Data
train_patient_rows = ismember(X100(:,end-1),train_patients_labels);
test_patient_rows  = ismember(X100(:,end-1),test_patients_labels);

train_patient = X100(train_patient_rows,:);
test_patient = X100(test_patient_rows,:);

% Preprocess for SVM
train_patient_features = train_patient(:,1:end-3);
train_patient_labels = train_patient(:,end);

test_patient_features = test_patient(:,1:end-3);
test_patient_labels = test_patient(:,end);

% Define Model
model_100 = fitcsvm(train_patient_features, train_patient_labels, 'KernelFunction', 'linear', ...
    'OptimizeHyperparameters', 'auto', ...
    'HyperparameterOptimizationOptions', struct('AcquisitionFunctionName', ...
    'expected-improvement-plus', 'ShowPlots', true));

% Test Model
result = predict(model_100, test_patient_features);
accuracy_100 = sum(result == test_patient_labels)/length(test_patient_labels)*100;

sp = sprintf("Test accuracy = %.2f", accuracy_100);
disp(sp);
time_100 = toc;








