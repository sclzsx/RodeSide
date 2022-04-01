function map_out = multiscale_gradient_cal(input,lowPassKS,alphaFact,beta,layers, out_dir)
    if ~exist(out_dir, 'dir')
        mkdir(out_dir)
    end

    pyramid_map = cell(2,layers);

    % Create pyramid for gradient map using central finite difference
%     Gk = fspecial('gaussian', lowPassKS, 1.0);
    Gk = fspecial('average', lowPassKS);
    gX = [-1, 0, 1];  % Central/Horizontal
    gY = [-1; 0; 1];  % Central/Vertical
    
    % Calculate gradients at various scales (bottom->top)
    Lt = input;
    for i = 1:layers
    	Gx = imfilter(Lt, gX, 'same') ./ double(2.0^(i));
        Gy = imfilter(Lt, gY, 'same') ./ double(2.0^(i));
        pyramid_map{1,i} = Gx;
        pyramid_map{2,i} = Gy;
        if i < layers  % Last downsampling is unnecessary for it is not used.
            Lt = imresize(imfilter(Lt, Gk, 'same'),0.5,'bilinear');% Downsampling
        end
    end
    
    % Find alpha factor which dets. the gradient mag. to remain as is.
    sqG2  = sqrt(pyramid_map{1,1}.^2 + pyramid_map{2,1}.^2); % Calculate it at the base level
    avGr  = mean(sqG2(:));
    alpha = avGr*alphaFact;
    
    phiKp1 = attenuation_mask(pyramid_map{1,layers}, pyramid_map{2,layers},alpha,beta);  % Start at the coarsest
    imwrite(phiKp1,[out_dir , num2str(layers), '_layer.jpg']);
    
    for i = layers-1 :-1:1
        [r, c] = size(pyramid_map{1,i});
        phiK   = attenuation_mask(pyramid_map{1,i}, pyramid_map{2,i},alpha,beta);
        imwrite(phiK,[out_dir, num2str(i),'_layer.jpg']);
        phiKp1 = imresize(phiKp1, [r, c], 'bilinear').*phiK;    
    end
    map_out = phiKp1;    
end



function  mask = attenuation_mask(Gx,Gy,alpha,beta)
        gradNorm = sqrt(Gx.^2+Gy.^2);  % L2-norm-Magnitude
        mask     = (alpha./gradNorm).*((gradNorm./alpha).^beta);
        mask(gradNorm==0) = 1;
end

