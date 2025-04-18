function output = energy(image)
    % function output = energy(image)
    % image = input image
    % output = energy of image
    [M,N] = size(image);
    square_values = image.^2;
    summed_values = sum(square_values(:,:),'all');
    output = summed_values/(M*N);
end