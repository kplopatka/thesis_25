function [R,G,B] = deconstruct(image)
    % Function Decomposes an RGB into its R, G, and B

    R(:,:) = image(:,:,1);
    G(:,:) = image(:,:,2);
    B(:,:) = image(:,:,3);
    
    R = uint8(R);
    G = uint8(G);
    B = uint8(B);

end