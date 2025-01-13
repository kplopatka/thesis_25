function feature_vector = get_feature_vector(image, level)    

    % Storing Coefficients
    approximation = cell(1,level);
    horizantal = cell(1,level);
    vertical = cell(1,level);
    diagonal = cell(1,level);
    
    % ------------------------------
    % Relevant Features
    mean_cell = cell(level,4);
    sigma_cell = cell(level,4);
    skewness_cell = cell(level,4);
    kurtosis_cell = cell(level,4);
    entropy_cell = cell(level,4);
    energy_cell = cell(level,4);
    RMS_cell = cell(level,4);
    meanAD_cell = cell(level,4);
    medianAD_cell = cell(level,4);
    
    % Row --> Corresponding Level
    % Col 1 = Approximation
    % Col 2 = Horizantal
    % Col 3 = Vertical 
    % Col 4 = Diagonal
    
    % ------------------------------
    % Grayscale the Image
    image_gray = rgb2gray(image);
    
    % Collect Coefficients at Each Decomposition Level
    for i = 1:level
    
        % Wavelet Decomposition
        [c, s] = wavedec2(image_gray,level,'haar');
        
        % Matlab Function to extract Coefficients at Specific Level
        [horiz, vert, diag] = detcoef2('all',c,s,i);
        approx = appcoef2(c,s,'haar',i);
        
        % Store in cells
        approximation{i} = approx(:,:);
        horizantal{i} = horiz(:,:);
        vertical{i} = vert(:,:);
        diagonal{i} = diag(:,:);
    
        % Mean
        mean_cell{i,1} = mean(approximation{i}, 'all');
        mean_cell{i,2} = mean(horizantal{i}, 'all');
        mean_cell{i,3} = mean(vertical{i}, 'all');
        mean_cell{i,4} = mean(diagonal{i}, 'all');
    
        % Variance
        sigma_cell{i,1} = var(approximation{i},1,'all');
        sigma_cell{i,2} = var(horizantal{i},1,'all');
        sigma_cell{i,3} = var(vertical{i},1,'all');
        sigma_cell{i,4} = var(diagonal{i},1,'all');
    
        % Skewness
        skewness_cell{i,1} = skewness(approximation{i},1,'all');
        skewness_cell{i,2} = skewness(horizantal{i},1,'all');
        skewness_cell{i,3} = skewness(vertical{i},1,'all');
        skewness_cell{i,4} = skewness(diagonal{i},1,'all');
    
        % Kurtosis
        kurtosis_cell{i,1} = kurtosis(approximation{i},1,'all');
        kurtosis_cell{i,2} = kurtosis(horizantal{i},1,'all');
        kurtosis_cell{i,3} = kurtosis(vertical{i},1,'all');
        kurtosis_cell{i,4} = kurtosis(diagonal{i},1,'all');
        
        % Entropy
        entropy_cell{i,1} = entropy(approximation{i});
        entropy_cell{i,2} = entropy(horizantal{i});
        entropy_cell{i,3} = entropy(vertical{i});
        entropy_cell{i,4} = entropy(diagonal{i});
    
        % Energy
        energy_cell{i,1} = energy(approximation{i});
        energy_cell{i,2} = energy(horizantal{i});
        energy_cell{i,3} = energy(vertical{i});
        energy_cell{i,4} = energy(diagonal{i});
        
        % RMS
        RMS_cell{i,1} = rms(approximation{i},'all');
        RMS_cell{i,2} = rms(horizantal{i},'all');
        RMS_cell{i,3} = rms(vertical{i},'all');
        RMS_cell{i,4} = rms(diagonal{i},'all');
    
        % Mean Absolute Deviation
        meanAD_cell{i,1} = mean(abs(approximation{i} - mean(approximation{i}, 'all')),'all');
        meanAD_cell{i,2} = mean(abs(horizantal{i} - mean(horizantal{i}, 'all')),'all');
        meanAD_cell{i,3} = mean(abs(vertical{i} - mean(vertical{i}, 'all')),'all');
        meanAD_cell{i,4} = mean(abs(diagonal{i} - mean(diagonal{i}, 'all')),'all');
    
        % Median Absolute Deviation
        medianAD_cell{i,1} = median(abs(approximation{i} - median(approximation{i}, 'all')),'all');
        medianAD_cell{i,2} = median(abs(horizantal{i} - median(horizantal{i}, 'all')),'all');
        medianAD_cell{i,3} = median(abs(vertical{i} - median(vertical{i}, 'all')),'all');
        medianAD_cell{i,4} = median(abs(diagonal{i} - median(diagonal{i}, 'all')),'all');
        
    end
    
    % Generate Feature Vector
    feature_vector = [mean_cell; sigma_cell; skewness_cell; kurtosis_cell;
                      entropy_cell; energy_cell; RMS_cell; meanAD_cell; 
                      medianAD_cell];
    
    % Transform to Matrix
    feature_vector = cell2mat(feature_vector);

    % Dimensions
    [M,N] = size(feature_vector);

    % Reshape
    feature_vector = reshape(feature_vector, 1, M * N);
      
    % Expect each feature vector to be of size (1, 1 + (level * 9 * 4))

    % Add space for relevant labels
    feature_vector(end+1) = 0; % For Mag Level
    feature_vector(end+1) = 0; % For Patient Number
    feature_vector(end+1) = 0; % For Benign / Malignant

end
    
    
    
    
    
    
    
    

