function [N]=ERGAS(MS,F)
MS=double(MS);
F=double(F);
[m,n,p]=size(F);
M1=F(:,:,1);
M2=F(:,:,2);
M3=F(:,:,3);
A1=reshape(M1,[m*n,1]);
A2=reshape(M2,[m*n,1]);
A3=reshape(M3,[m*n,1]);
C1=(sum(sum((MS(:,:,1)-F(:,:,1)).^2))/(m*n))^(1/2);
C2=(sum(sum((MS(:,:,2)-F(:,:,2)).^2))/(m*n))^(1/2);
C3=(sum(sum((MS(:,:,3)-F(:,:,3)).^2))/(m*n))^(1/2);
S1=(C1/mean(A1))^2;
S2=(C2/mean(A2))^2;
S3=(C3/mean(A3))^2;
if p==4
    M4=F(:,:,4);
    A4=reshape(M4,[m*n,1]);
    C4=(sum(sum((MS(:,:,4)-F(:,:,4)).^2))/(m*n))^(1/2);
    S4=(C4/mean(A4))^2;
    S=S1+S2+S3+S4;
    N=((S/4)^(1/2))*(25);
else
    S=S1+S2+S3;
    N=((S/3)^(1/2))*(25);
end
