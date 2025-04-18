function feature_vector = hsv_vector(image, level, type)    

    % Output Feature Vector is horizantally concatenated to read as such:
    % [Mean, Var, ... MeanAD];
    % [Mean(Approx1-level, Horiz1-level, Vert1-level, Diag1-level), Var(Approx1-7 ...
    % Each statistical parameter type can be found by 4*levels
    % --> i.e. (levels = 7) Mean = elements 1-28, Var = elements 29-56 ...

    % Storing Coefficients
    approximation = cell(1,level);
    horizantal = cell(1,level);
    vertical = cell(1,level);
    diagonal = cell(1,level);
    
    % ------------------------------
    % Relevant Features
    mean_v = zeros(level,4);
    sigma_v = zeros(level,4);
    skewness_v = zeros(level,4);
    kurtosis_v = zeros(level,4);
    entropy_v = zeros(level,4);
    energy_v = zeros(level,4);
    RMS_v = zeros(level,4);
    meanAD_v = zeros(level,4);
    medianAD_v = zeros(level,4);
    
    % Row --> Corresponding Level
    % Col 1 = Approximation
    % Col 2 = Horizantal
    % Col 3 = Vertical 
    % Col 4 = Diagonal
    
    % ------------------------------
    % Deconstruct Image into HSV
    hsv_image = rgb2hsv(image);
    [H,S,V] = deconstruct(hsv_image);
    hsv_image = {H, S, V};
    start = 1;

    % For all three color layers...
    for a = 1:3
    % for a = 1:3
        % Collect Coefficients at Each Decomposition Level
        for i = 1:level
        
            % Wavelet Decomposition
            [c, s] = wavedec2(hsv_image{a},level,type);
            
            % Matlab Function to extract Coefficients at Specific Level
            [horiz, vert, diag] = detcoef2('all',c,s,i);
            approx = appcoef2(c,s,type,i);
            
            % Store in cells
            approximation{i} = approx(:,:);
            horizantal{i} = horiz(:,:);
            vertical{i} = vert(:,:);
            diagonal{i} = diag(:,:);
        
            % Mean
            mean_v(i,1) = mean(approximation{i},'all');
            mean_v(i,2) = mean(horizantal{i}, 'all');
            mean_v(i,3) = mean(vertical{i}, 'all');
            mean_v(i,4) = mean(diagonal{i}, 'all');
        
            % Variance
            sigma_v(i,1) = var(approximation{i},1,'all');
            sigma_v(i,2) = var(horizantal{i},1,'all');
            sigma_v(i,3) = var(vertical{i},1,'all');
            sigma_v(i,4) = var(diagonal{i},1,'all');
        
            % Skewness
            skewness_v(i,1) = skewness(approximation{i},1,'all');
            skewness_v(i,2) = skewness(horizantal{i},1,'all');
            skewness_v(i,3) = skewness(vertical{i},1,'all');
            skewness_v(i,4) = skewness(diagonal{i},1,'all');
        
            % Kurtosis
            kurtosis_v(i,1) = kurtosis(approximation{i},1,'all');
            kurtosis_v(i,2) = kurtosis(horizantal{i},1,'all');
            kurtosis_v(i,3) = kurtosis(vertical{i},1,'all');
            kurtosis_v(i,4) = kurtosis(diagonal{i},1,'all');
            
            % Entropy
            entropy_v(i,1) = entropy(approximation{i});
            entropy_v(i,2) = entropy(horizantal{i});
            entropy_v(i,3) = entropy(vertical{i});
            entropy_v(i,4) = entropy(diagonal{i});
        
            % Energy
            energy_v(i,1) = energy(approximation{i});
            energy_v(i,2) = energy(horizantal{i});
            energy_v(i,3) = energy(vertical{i});
            energy_v(i,4) = energy(diagonal{i});
            
            % RMS
            RMS_v(i,1) = rms(approximation{i},'all');
            RMS_v(i,2) = rms(horizantal{i},'all');
            RMS_v(i,3) = rms(vertical{i},'all');
            RMS_v(i,4) = rms(diagonal{i},'all');
        
            % Mean Absolute Deviation
            meanAD_v(i,1) = mean(abs(approximation{i} - mean(approximation{i}, 'all')),'all');
            meanAD_v(i,2) = mean(abs(horizantal{i} - mean(horizantal{i}, 'all')),'all');
            meanAD_v(i,3) = mean(abs(vertical{i} - mean(vertical{i}, 'all')),'all');
            meanAD_v(i,4) = mean(abs(diagonal{i} - mean(diagonal{i}, 'all')),'all');
        
            % Median Absolute Deviation
            medianAD_v(i,1) = median(abs(approximation{i} - median(approximation{i}, 'all')),'all');
            medianAD_v(i,2) = median(abs(horizantal{i} - median(horizantal{i}, 'all')),'all');
            medianAD_v(i,3) = median(abs(vertical{i} - median(vertical{i}, 'all')),'all');
            medianAD_v(i,4) = median(abs(diagonal{i} - median(diagonal{i}, 'all')),'all');
            
        end
        
        temp_feature_vector = [mean_v, sigma_v, skewness_v, kurtosis_v, entropy_v, energy_v, RMS_v, meanAD_v, medianAD_v];
    
        % Dimensions
        [M,N] = size(temp_feature_vector);
    
        % Reshape
        temp_feature_vector = reshape(temp_feature_vector, 1, M * N);
        
        % New Dimensions
        [X, Y] = size(temp_feature_vector);

        % Fill
        feature_vector(1,start:(start + Y - 1)) = temp_feature_vector(1,:);

        start = start + Y;
    end
        
    feature_vector(end+1) = 0; % For Mag Level
    feature_vector(end+1) = 0; % For Patient Number
    feature_vector(end+1) = 0; % For Benign / Malignant
    
end
    
function [R,G,B] = deconstruct(image)
    % Function Decomposes an RGB into its R, G, and B

    R(:,:) = image(:,:,1);
    G(:,:) = image(:,:,2);
    B(:,:) = image(:,:,3);
    
    %R = uint8(R);
    %G = uint8(G);
    %B = uint8(B);

end

function RGB_image = reconstruct(R,G,B)
    % Function Decomposes an RGB into its R, G, and B
    
    RGB_image(:,:,1) = R(:,:);
    RGB_image(:,:,2) = G(:,:);
    RGB_image(:,:,3) = B(:,:);

    RGB_image = uint8(RGB_image);
    
end
    
    
    
    
    
    

