%% Setup

close all;
clear all;


% Initialize Counter for Cells
benign_counter = 1;
malignant_counter = 1;

% Holds labeled feature vectors
benign_vectors = cell(2480,1);
malignant_vectors = cell(5429,1);

% Variables to access benign and malignant data information
benign_data = './BreaKHis_Mod/histology_slides/breast/benign/SOB';
malignant_data = './BreaKHis_Mod/histology_slides/breast/malignant/SOB';

% Store Each Subfolder of Benign and Malignant
benign_types = dir(benign_data);
benign_types = benign_types([benign_types.isdir]); % Keep only directories
benign_types = benign_types(~ismember({benign_types.name}, {'.', '..'})); % Exclude '.' and '..'

malignant_types = dir(malignant_data);
malignant_types = malignant_types([malignant_types.isdir]); % Keep only directories
malignant_types = malignant_types(~ismember({malignant_types.name}, {'.', '..'})); % Exclude '.' and '..'

% Initialize more counters
patient_num = 1;
total_single_patient_images = 0;

% Main Iteration through all datasets

%% For Benign...

tic;

% Iterate thropugh benign cancer types
for i = 1:length(benign_types)

    % Iterate through each type of benign tumor
    specific_benign_type = fullfile(benign_types(i).folder, benign_types(i).name);
    
    % Access SOB Types (level 2)
    SOB_types = dir(specific_benign_type);
    SOB_types = SOB_types([SOB_types.isdir]);
    SOB_types = SOB_types(~ismember({SOB_types.name}, {'.', '..'}));
    
    % Iterate through SOB Types
    for j = 1:length(SOB_types)
        specific_SOB_type = fullfile(SOB_types(j).folder, SOB_types(j).name);
        
        % Access Magnification Levels (level 3)
        magnification_levels = dir(specific_SOB_type);
        magnification_levels = magnification_levels([magnification_levels.isdir]);
        magnification_levels = magnification_levels(~ismember({magnification_levels.name}, {'.', '..'}));
        
        % Iterate through Magnification Levels

        % Will always iterate through in the following order
        % k = 1 --> 100X
        % k = 2 --> 200X
        % k = 3 --> 400X
        % k = 4 -->  40X

        for k = 1:length(magnification_levels)
            specific_mag_level = fullfile(magnification_levels(k).folder, magnification_levels(k).name);

            % Access all Images
            image_files = dir(fullfile(specific_mag_level, '*.png')); % Adjust file extension as needed
            for m = 1:length(image_files)
                
                % Full path to the image
                image_path = fullfile(image_files(m).folder, image_files(m).name);
                
                % Process the image
                img = imread(image_path);
            
                % Single feature vector
                vector = rgb_vector(img,7,'sym4');
            
                % Add a magnification label to the vector
                vector(end-2) = k;
            
                % Store in Benign Cell
                benign_vectors{benign_counter,1} = vector;
            
                % Increment Counter
                benign_counter = benign_counter + 1;
            
            end

            total_single_patient_images = total_single_patient_images + length(image_files);
        end
        
        % These counters help append patient labels
        start_val = benign_counter - total_single_patient_images;
        end_val = start_val + total_single_patient_images - 1;

        % Append patient label to end of the feature vector
        for z = start_val:end_val
            benign_vectors{z,1}(:,end-1) = patient_num;
        end

        % This counter keeps track of which patient is which
        patient_num = patient_num + 1;
        total_single_patient_images = 0;

    end

end

% Benign Label = -1
benign_vectors = cell2mat(benign_vectors);
benign_vectors(:,end) = -1;

benign_time = toc;

%% For Malignant...

tic;

% Iterate thropugh malignant cancer types
for i = 1:length(malignant_types)
    specific_benign_type = fullfile(malignant_types(i).folder, malignant_types(i).name);
    
    % Access SOB Types (level 2)
    SOB_types = dir(specific_benign_type);
    SOB_types = SOB_types([SOB_types.isdir]);
    SOB_types = SOB_types(~ismember({SOB_types.name}, {'.', '..'}));
    
    % Iterate through SOB Types
    for j = 1:length(SOB_types)
        specific_SOB_type = fullfile(SOB_types(j).folder, SOB_types(j).name);
        
        % Access Magnification Levels (level 3)
        magnification_levels = dir(specific_SOB_type);
        magnification_levels = magnification_levels([magnification_levels.isdir]);
        magnification_levels = magnification_levels(~ismember({magnification_levels.name}, {'.', '..'}));
        
        % Iterate through Magnification Levels

        % Will always iterate through in the following order
        % k = 1 --> 100X
        % k = 2 --> 200X
        % k = 3 --> 400X
        % k = 4 -->  40X

        for k = 1:length(magnification_levels)
            specific_mag_level = fullfile(magnification_levels(k).folder, magnification_levels(k).name);
            
            % Access all Images
            image_files = dir(fullfile(specific_mag_level, '*.png')); % Adjust file extension as needed
            for m = 1:length(image_files)
                
                % Full path to the image
                image_path = fullfile(image_files(m).folder, image_files(m).name);
                
                % Process the image
                img = imread(image_path);

                % Single feature vector
                vector = rgb_vector(img,7,'sym4');

                % Add a magnification label to the vector
                vector(end-2) = k;

                % Store in Benign Cell
                malignant_vectors{malignant_counter} = vector;

                % Increment Counter
                malignant_counter = malignant_counter + 1;

            end

            total_single_patient_images = total_single_patient_images + length(image_files);

        end
   
        % These counters help append patient labels
        start_val = malignant_counter - total_single_patient_images;
        end_val = start_val + total_single_patient_images - 1;

        % Append patient label to end of the feature vector
        for z = start_val:end_val
            malignant_vectors{z,1}(:,end-1) = patient_num;
        end

        % This counter keeps track of which patient is which
        patient_num = patient_num + 1;
        total_single_patient_images = 0;

    end

end

% Malignant Label = 1
malignant_vectors = cell2mat(malignant_vectors);
malignant_vectors(:,end) = 1;
malignant_time = toc;

%% Post-Processing
gray_sym4_vector = [malignant_vectors; benign_vectors];
    
sum_time = (malignant_time + benign_time)/60;

minutes = floor(sum_time)
sec = (sum_time - minutes)*60





