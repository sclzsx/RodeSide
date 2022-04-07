clc;
clear;

use_mask = 0;

rgb = imread('Data2/6.JPG');
sob = imread('Data2/61.png');

sob(sob>0) = 1;
sob = double(sob);

sob_ori = sob;

[H,W] = size(sob);
        
if use_mask
        eve = even_light(rgb);
        % eve = rgb;
        
        [h,s,v] = rgb2hsv(eve);
        
        [H,W] = size(h);
        hue_mask = zeros([H W]);
        for i=1:H
            for j=1:W
                tem = h(i,j);
                if tem<0.25
                    hue_mask(i,j) = 13.6 * tem * tem;
                elseif tem < 0.42
                    hue_mask(i,j) = -18.5 * (tem-0.3333) * (tem-0.3333) + 1;
                else
                    hue_mask(i,j) = 0.003624 * (tem ^ -6.29);
                end
            end
        end
        
        % hue_mask = hue_mask .* 0.3;
        
        hue_mask = imresize(hue_mask, 0.5);
        
        se = strel('disk',7);
        hue_mask = imdilate(hue_mask,se);
        
        hue_mask = max(hue_mask - 0.05, 0) .^ 0.5;
        
        hue_mask = imresize(hue_mask, [H,W]);
        
        hue_mask = 1- hue_mask;
        hue_mask(hue_mask<0.4) = 0;
        hue_mask(hue_mask>=0.4) = 1;
        
        
        
        sob = sob.* hue_mask;
        
        draw_mask = ones([H, W]);
        draw_mask(1:floor(H/4), :) = 0;
        draw_mask(:, 1:floor(H/10)) = 0;
        draw_mask(:, W - floor(H/10):end) = 0;
        
        sob = sob .* draw_mask;
end

%%

gausFilter = fspecial('gaussian', [3,3], 1);
gau = imfilter(double(sob), gausFilter);
% 
% sob(sob<0.001) = 0;
% 
sob(sob>0) = 1;
% 
% sob = sob.* hue_mask;
% 
% gau = sob;

se = strel('disk',5);
sob = imdilate(gau,se);
% 
% se = strel('disk',3);
% gau2 = imerode(gau,se);
% 
% gau3 = gau1 - gau2;
% 
% sob = sob .* gau3;



%%

flag = 0;
top_y = 0;
top_x_left = 0;
for i=1:H
    for j=1:W
        if sob(i,j)>0
            top_y = i;
            top_x_left = j;
            flag = 1;
        end
        if flag ==1
            break;
        end
    end
    if flag ==1
        break;
    end
end

top_x_right = 0;
for j=W:-1:1
    if sob(top_y,j) > 0
        top_x_right = j;
        break;
    end
end


flag = 0;
bot_y = 0;
bot_x_left = 0;
for i=H:-1:1
    for j=1:W
        if sob(i,j)>0
            bot_y = i;
            bot_x_left = j;
            flag = 1;
        end
        if flag ==1
            break;
        end
    end
    if flag ==1
        break;
    end
end

bot_x_right = 0;
for j=W:-1:1
    if sob(bot_y,j) > 0
        bot_x_right = j;
        break;
    end
end

top_x = floor((top_x_left + top_x_right) /2);
bot_x = floor((bot_x_left + bot_x_right) /2);

% imshow(sob)
% plot([top_x,top_y],[bot_x,bot_y])

% y = ((bot_y - top_y)/(bot_x-top_x))*(x-top_x) + top_y;

left_mask = zeros([H,W]);
right_mask = zeros([H,W]);
for i=1:H
    for j=1:W
        x = j;
        y = ((bot_y - top_y)/(bot_x-top_x))*(x -top_x) + top_y;
        if y < i
            left_mask(i,j) = 1;
        else
            right_mask(i,j) = 1;
        end
    end
end



sob_left = sob .* left_mask;
sob_right = sob .* right_mask;


                L = bwconncomp(sob_left);
                S = regionprops(L, 'Area');
                Ss = [];
                for i=1:length(S)
                    s = S(i).Area;
                    Ss = [Ss s];
                end
                s = mean(Ss);
                idx = find([S.Area] > s);
                sob_left = ismember(labelmatrix(L),idx);
                
                L = bwconncomp(sob_right);
                S = regionprops(L, 'Area');
                Ss = [];
                for i=1:length(S)
                    s = S(i).Area;
                    Ss = [Ss s];
                end
                s = mean(Ss);
                idx = find([S.Area] > s);
                sob_right = ismember(labelmatrix(L),idx);

                sob_left = sob_ori .* sob_left;
                sob_right = sob_ori .* sob_right;

left_X = [];
left_Y = [];
right_X = [];
right_Y = [];

for i=1:W
    flag = 0;
    for j=1:H
        if flag
            break;
        end
        if sob_left(j,i) > 0
            left_X = [left_X i];
            left_Y = [left_Y j];
            flag = 1;
        end
    end
end

for i=1:W
    flag = 0;
    for j=1:H
        if flag
            break;
        end
        if sob_right(j,i) > 0
            right_X = [right_X i];
            right_Y = [right_Y j];
            flag = 1;
        end
    end
end

% plot(left_X,left_Y);
% plot(right_X, right_Y);

% show = sob_left + gray + sob_right;

coefficient = polyfit(left_X,left_Y,3);
left_Y2 = polyval(coefficient,left_X);

coefficient = polyfit(right_X,right_Y,3);
right_Y2 = polyval(coefficient,right_X,3);

left_X = floor(left_X);
left_Y2 = floor(abs(left_Y2));
right_X = floor(right_X);
right_Y2 = floor(abs(right_Y2));
plot(left_X, -left_Y2);
hold on;
plot(right_X, -right_Y2);

dlmwrite('r_X.txt',right_X);
dlmwrite('r_Y.txt',right_Y2);
dlmwrite('l_X.txt',left_X);
dlmwrite('l_Y.txt',left_Y2);

% % % eps = 0.000001;
% % % I_log = I_filt;
% % % 
% % layer = 5;
% % lowPassKS = 5;
% % alpha = 0.2; % Will multiplied by the average grad. magnitude  
% % beta  = 0.85; % Attenuates/Amplifies larger/smaller magnitudes
% % gradient_mask = multiscale_gradient_cal(mea,lowPassKS,alpha,beta,layer, './');
% % gradient_mask = gradient_mask.*gradient_mask;
% % img_mask = 1 - gradient_mask ;
% % img_mask = min(1,max(img_mask,0));
% % 
% % base = 0.2;
% % t = 2;
% % img_mask_n = max(img_mask - base,0);
% % img_mask_n = power(img_mask_n, t);
% % smooth_mask = 1-img_mask_n;
% 
% 
% % hf = [0, 0, 0;
% %       1, 0, -1;
% %      0, 0, 0];
% % vf = [0, 1, 0;
% %       0, 0, 0;
% %       0, -1, 0];
% % df1 = [1, 0, 0;
% %        0, 0, 0;
% %        0, 0, -1];
% % df2 = [0, 0, 1;
% %        0, 0, 0;
% %        -1, 0, 0];
% % agv = abs(imfilter(luma, vf));
% % agh = abs(imfilter(luma, hf));
% % agz = abs(imfilter(luma, df1));
% % agn = abs(imfilter(luma, df2));
% % 
% % ed1_metric = max(max(max(agv, agh), agz), agn);
% % ed2_metric = min(max(ed1_metric./max(luma, 1.0), 0.0), 1.0);