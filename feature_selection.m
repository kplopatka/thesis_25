%% Loading Relevant Wavelet Features
clear;

% Import Features
load("new_haar_vectors.mat");
malignant_vectors = new_haar_vectors(1:5429,:);
benign_vectors = new_haar_vectors(5430:end,:);

% Check Relevant Features
[M,N] = size(benign_vectors);
[X,Y] = size(malignant_vectors);

% Varies by Level of Decomp
step_size = (N-3)/9;

% Features
mean_b = benign_vectors(:,1:step_size);
mean_m = malignant_vectors(:,1:step_size);

sigma_b = benign_vectors(:,(step_size+1):2*step_size);
sigma_m = malignant_vectors(:,(step_size+1):2*step_size);

skew_b = benign_vectors(:,(2*step_size+1):3*step_size);
skew_m = malignant_vectors(:,(2*step_size+1):3*step_size);

kurt_b = benign_vectors(:,(3*step_size+1):4*step_size);
kurt_m = malignant_vectors(:,(3*step_size+1):4*step_size);

entropy_b = benign_vectors(:,(4*step_size+1):5*step_size);
entropy_m = malignant_vectors(:,(4*step_size+1):5*step_size);

energy_b = benign_vectors(:,(5*step_size+1):6*step_size);
energy_m = malignant_vectors(:,(5*step_size+1):6*step_size);

RMS_b = benign_vectors(:,(6*step_size+1):7*step_size);
RMS_m = malignant_vectors(:,(6*step_size+1):7*step_size);

meanAD_b = benign_vectors(:,(7*step_size+1):8*step_size);
meanAD_m = malignant_vectors(:,(7*step_size+1):8*step_size);

medAD_b = benign_vectors(:,(8*step_size+1):9*step_size);
medAD_m = malignant_vectors(:,(8*step_size+1):9*step_size);

%% Approximation Histograms
for i = 1:7
    figure(i)
    
    subplot(331)
    histogram(mean_b(:,i),75)
    grid on;
    xlabel('Mean Value')
    ylabel('Number of Samples in Bin')
    title('Feature: Mean')
    hold on;
    histogram(mean_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    subplot(332)
    histogram(sigma_b(:,i),75)
    grid on;
    xlabel('Sigma Value')
    ylabel('Number of Samples in Bin')
    title('Feature: Standard Deviation')
    hold on;
    histogram(sigma_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    subplot(333)
    histogram(skew_b(:,i),75)
    grid on;
    xlabel('Skewness Value')
    ylabel('Number of Samples in Bin')
    title('Feature: Skewness')
    hold on;
    histogram(skew_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    subplot(334)
    histogram(kurt_b(:,i),75)
    grid on;
    xlabel('Kurtosis Value')
    ylabel('Number of Samples in Bin')
    title('Feature: Kurtosis')
    hold on;
    histogram(kurt_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    subplot(335)
    histogram(entropy_b(:,i),75)
    grid on;
    xlabel('Entropy Value')
    ylabel('Number of Samples in Bin')
    title('Feature: Entropy')
    hold on;
    histogram(entropy_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    subplot(336)
    histogram(energy_b(:,i),75)
    grid on;
    xlabel('Energy Value')
    ylabel('Number of Samples in Bin')
    title('Feature: Energy')
    hold on;
    histogram(energy_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    subplot(337)
    histogram(RMS_b(:,i),75)
    grid on;
    xlabel('RMS Value')
    ylabel('Number of Samples in Bin')
    title('Feature: RMS')
    hold on;
    histogram(RMS_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    subplot(338)
    histogram(meanAD_b(:,i),75)
    grid on;
    xlabel('Mean Average Deviation Value')
    ylabel('Number of Samples in Bin')
    title('Feature: Mean Average Deviation')
    hold on;
    histogram(meanAD_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    subplot(339)
    histogram(medAD_b(:,i),75)
    grid on;
    xlabel('Median Average Deviation Value')
    ylabel('Number of Samples in Bin')
    title('Feature: Median Average Deviation')
    hold on;
    histogram(medAD_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    sgtitle(['Wavelet Decomposition Level: ' num2str(i)]);

end

%% Horizantal Histograms

for i = 8:14
    figure(i)
    
    subplot(331)
    histogram(mean_b(:,i),75)
    grid on;
    xlabel('Mean Value')
    ylabel('Number of Samples in Bin')
    title('Feature: Mean')
    hold on;
    histogram(mean_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    subplot(332)
    histogram(sigma_b(:,i),75)
    grid on;
    xlabel('Sigma Value')
    ylabel('Number of Samples in Bin')
    title('Feature: Standard Deviation')
    hold on;
    histogram(sigma_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    subplot(333)
    histogram(skew_b(:,i),75)
    grid on;
    xlabel('Skewness Value')
    ylabel('Number of Samples in Bin')
    title('Feature: Skewness')
    hold on;
    histogram(skew_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    subplot(334)
    histogram(kurt_b(:,i),75)
    grid on;
    xlabel('Kurtosis Value')
    ylabel('Number of Samples in Bin')
    title('Feature: Kurtosis')
    hold on;
    histogram(kurt_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    subplot(335)
    histogram(entropy_b(:,i),75)
    grid on;
    xlabel('Entropy Value')
    ylabel('Number of Samples in Bin')
    title('Feature: Entropy')
    hold on;
    histogram(entropy_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    subplot(336)
    histogram(energy_b(:,i),75)
    grid on;
    xlabel('Energy Value')
    ylabel('Number of Samples in Bin')
    title('Feature: Energy')
    hold on;
    histogram(energy_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    subplot(337)
    histogram(RMS_b(:,i),75)
    grid on;
    xlabel('RMS Value')
    ylabel('Number of Samples in Bin')
    title('Feature: RMS')
    hold on;
    histogram(RMS_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    subplot(338)
    histogram(meanAD_b(:,i),75)
    grid on;
    xlabel('Mean Average Deviation Value')
    ylabel('Number of Samples in Bin')
    title('Feature: Mean Average Deviation')
    hold on;
    histogram(meanAD_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    subplot(339)
    histogram(medAD_b(:,i),75)
    grid on;
    xlabel('Median Average Deviation Value')
    ylabel('Number of Samples in Bin')
    title('Feature: Median Average Deviation')
    hold on;
    histogram(medAD_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    sgtitle(['Wavelet Decomposition Level: ' num2str(i-7)]);

end

%% Vertical Histograms

for i = 15:21
    figure(i)
    
    subplot(331)
    histogram(mean_b(:,i),75)
    grid on;
    xlabel('Mean Value')
    ylabel('Number of Samples in Bin')
    title('Feature: Mean')
    hold on;
    histogram(mean_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    subplot(332)
    histogram(sigma_b(:,i),75)
    grid on;
    xlabel('Sigma Value')
    ylabel('Number of Samples in Bin')
    title('Feature: Standard Deviation')
    hold on;
    histogram(sigma_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    subplot(333)
    histogram(skew_b(:,i),75)
    grid on;
    xlabel('Skewness Value')
    ylabel('Number of Samples in Bin')
    title('Feature: Skewness')
    hold on;
    histogram(skew_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    subplot(334)
    histogram(kurt_b(:,i),75)
    grid on;
    xlabel('Kurtosis Value')
    ylabel('Number of Samples in Bin')
    title('Feature: Kurtosis')
    hold on;
    histogram(kurt_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    subplot(335)
    histogram(entropy_b(:,i),75)
    grid on;
    xlabel('Entropy Value')
    ylabel('Number of Samples in Bin')
    title('Feature: Entropy')
    hold on;
    histogram(entropy_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    subplot(336)
    histogram(energy_b(:,i),75)
    grid on;
    xlabel('Energy Value')
    ylabel('Number of Samples in Bin')
    title('Feature: Energy')
    hold on;
    histogram(energy_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    subplot(337)
    histogram(RMS_b(:,i),75)
    grid on;
    xlabel('RMS Value')
    ylabel('Number of Samples in Bin')
    title('Feature: RMS')
    hold on;
    histogram(RMS_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    subplot(338)
    histogram(meanAD_b(:,i),75)
    grid on;
    xlabel('Mean Average Deviation Value')
    ylabel('Number of Samples in Bin')
    title('Feature: Mean Average Deviation')
    hold on;
    histogram(meanAD_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    subplot(339)
    histogram(medAD_b(:,i),75)
    grid on;
    xlabel('Median Average Deviation Value')
    ylabel('Number of Samples in Bin')
    title('Feature: Median Average Deviation')
    hold on;
    histogram(medAD_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    sgtitle(['Wavelet Decomposition Level: ' num2str(i-14)]);

end
    
%% Diagonal Histograms

for i = 22:28
    figure(i)
    
    subplot(331)
    histogram(mean_b(:,i),75)
    grid on;
    xlabel('Mean Value')
    ylabel('Number of Samples in Bin')
    title('Feature: Mean')
    hold on;
    histogram(mean_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    subplot(332)
    histogram(sigma_b(:,i),75)
    grid on;
    xlabel('Sigma Value')
    ylabel('Number of Samples in Bin')
    title('Feature: Standard Deviation')
    hold on;
    histogram(sigma_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    subplot(333)
    histogram(skew_b(:,i),75)
    grid on;
    xlabel('Skewness Value')
    ylabel('Number of Samples in Bin')
    title('Feature: Skewness')
    hold on;
    histogram(skew_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    subplot(334)
    histogram(kurt_b(:,i),75)
    grid on;
    xlabel('Kurtosis Value')
    ylabel('Number of Samples in Bin')
    title('Feature: Kurtosis')
    hold on;
    histogram(kurt_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    subplot(335)
    histogram(entropy_b(:,i),75)
    grid on;
    xlabel('Entropy Value')
    ylabel('Number of Samples in Bin')
    title('Feature: Entropy')
    hold on;
    histogram(entropy_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    subplot(336)
    histogram(energy_b(:,i),75)
    grid on;
    xlabel('Energy Value')
    ylabel('Number of Samples in Bin')
    title('Feature: Energy')
    hold on;
    histogram(energy_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    subplot(337)
    histogram(RMS_b(:,i),75)
    grid on;
    xlabel('RMS Value')
    ylabel('Number of Samples in Bin')
    title('Feature: RMS')
    hold on;
    histogram(RMS_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    subplot(338)
    histogram(meanAD_b(:,i),75)
    grid on;
    xlabel('Mean Average Deviation Value')
    ylabel('Number of Samples in Bin')
    title('Feature: Mean Average Deviation')
    hold on;
    histogram(meanAD_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    subplot(339)
    histogram(medAD_b(:,i),75)
    grid on;
    xlabel('Median Average Deviation Value')
    ylabel('Number of Samples in Bin')
    title('Feature: Median Average Deviation')
    hold on;
    histogram(medAD_m(:,i), 75);
    legend('Benign','Malignant')
    hold off;

    sgtitle(['Wavelet Decomposition Level: ' num2str(i-21)]);

end


