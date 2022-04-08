

clear all;
close all;
clc;

rgb = imread('Data2/3.JPG');
rgb = double(rgb)/255;

R=rgb(:,:,1);
G=rgb(:,:,2);
B=rgb(:,:,3);

[H,S,V] = my_rgb2hsv(R,G,B);
% [R,G,B] = my_hsv2rgb(H,S,V);

[h,s,v] = rgb2hsv(rgb);
% [r,g,b] = hsv2rgb(h,s,v);

%%
% function [H,S,V] = my_rgb2hsv(R,G,B)
%     [m,n] = size(R);
%     %%图像的RGB2HSV
%     H=zeros(m,n);   %色相角
%     S=zeros(m,n);   %饱和度
%     V=zeros(m,n);   %明度
%     for i=1:m
%        for j=1:n
%            r=R(i,j);
%            g=G(i,j);
%            b=B(i,j);
%            MAX=max([r,g,b]);
%            MIN=min([r,g,b]);
% 
%            if MAX==MIN
%                 H(i,j)=0;
%            elseif MAX==r && g>=b
%                 H(i,j)=60*(g-b)/(MAX-MIN);
%            elseif MAX==r && g<b
%                 H(i,j)=60*(g-b)/(MAX-MIN)+360;
%            elseif MAX==g
%                 H(i,j)=60*(b-r)/(MAX-MIN)+120;
%            elseif MAX==b
%                 H(i,j)=60*(r-g)/(MAX-MIN)+240;
%            end
% 
%            if MAX==0
%                 S(i,j)=0;
%            else
%                 S(i,j)=1-MIN/MAX;
%            end
% 
%            V(i,j)=MAX;
%        end
%     end
%     H = double(H) / 360;
% end

% function [R,G,B] = my_hsv2rgb(H,S,V)
%     H = double(H) * 360;
%     [m,n] = size(H);
%     %%图像的RGB2HSV
%     R=zeros(m,n);   %色相角
%     G=zeros(m,n);   %饱和度
%     B=zeros(m,n);   %明度
%     
%     %%图像HSV2RGB
%     for i=1:m
%         for j=1:n
%             h=floor(H(i,j)/60);
%             f=H(i,j)/60-h;
%             v=V(i,j);
%             s=S(i,j);
%             p=v*(1-s);
%             q=v*(1-f*s);
%             t=v*(1-(1-f)*s);
% 
%             if h==0
%                 R(i,j)=v;G(i,j)=t;B(i,j)=p;
%             elseif h==1
%                 R(i,j)=q;G(i,j)=v;B(i,j)=p;            
%             elseif h==2
%                 R(i,j)=p;G(i,j)=v;B(i,j)=t;            
%             elseif h==3
%                 R(i,j)=p;G(i,j)=q;B(i,j)=v;            
%             elseif h==4
%                 R(i,j)=t;G(i,j)=p;B(i,j)=v;            
%             elseif h==5
%                 R(i,j)=v;G(i,j)=p;B(i,j)=q;            
%             end
%         end
%     end
% end



