function RGB_image = reconstruct(R,G,B)
    % Function Decomposes an RGB into its R, G, and B
    
    RGB_image(:,:,1) = R(:,:);
    RGB_image(:,:,2) = G(:,:);
    RGB_image(:,:,3) = B(:,:);

    RGB_image = uint8(RGB_image);
    
end