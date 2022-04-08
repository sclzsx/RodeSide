function [R,G,B] = my_hsv2rgb(H,S,V)
    H = double(H) * 360;
    [m,n] = size(H);
    %%图像的RGB2HSV
    R=zeros(m,n);   %色相角
    G=zeros(m,n);   %饱和度
    B=zeros(m,n);   %明度
    
    %%图像HSV2RGB
    for i=1:m
        for j=1:n
            h=floor(H(i,j)/60);
            f=H(i,j)/60-h;
            v=V(i,j);
            s=S(i,j);
            p=v*(1-s);
            q=v*(1-f*s);
            t=v*(1-(1-f)*s);

            if h==0
                R(i,j)=v;G(i,j)=t;B(i,j)=p;
            elseif h==1
                R(i,j)=q;G(i,j)=v;B(i,j)=p;            
            elseif h==2
                R(i,j)=p;G(i,j)=v;B(i,j)=t;            
            elseif h==3
                R(i,j)=p;G(i,j)=q;B(i,j)=v;            
            elseif h==4
                R(i,j)=t;G(i,j)=p;B(i,j)=v;            
            elseif h==5
                R(i,j)=v;G(i,j)=p;B(i,j)=q;            
            end
        end
    end
end