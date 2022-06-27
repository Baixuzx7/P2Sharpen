function [N]=RASE(MS,F)
MS=double(MS);
F=double(F);

[m,n,p]=size(F);
M1=F(:,:,1);
M2=F(:,:,2);
M3=F(:,:,3);

A1=reshape(M1,[m*n,1]);
A2=reshape(M2,[m*n,1]);
A3=reshape(M3,[m*n,1]);

C1=(sum(sum((MS(:,:,1)-F(:,:,1)).^2))/(m*n));
C2=(sum(sum((MS(:,:,2)-F(:,:,2)).^2))/(m*n));
C3=(sum(sum((MS(:,:,3)-F(:,:,3)).^2))/(m*n));


if p==4
    M4=F(:,:,4);
    A4=reshape(M4,[m*n,1]);
    C4=(sum(sum((MS(:,:,4)-F(:,:,4)).^2))/(m*n));
    C=C1+C2+C3+C4;
    N=((C/4)^(1/2))*100/(mean(mean(mean(MS))));
else
    C=C1+C2+C3;
    N=((C/3)^(1/2))*100/(mean(mean(mean(MS))));
end
