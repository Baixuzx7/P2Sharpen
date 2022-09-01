function [T]=QAVE(MS,PANMS)
MS=double(MS);
PANMS=double(PANMS);
[n,m,d]=size(PANMS);
MS=MS(:,:,1:d);

MX= mean(MS,3);
MY= mean(PANMS,3);


% Each value minus the mean along each band
M1=MS(:,:,1)-MX;
M2=MS(:,:,2)-MX;
M3=MS(:,:,3)-(MX);

P1=PANMS(:,:,1)-(MY);
P2=PANMS(:,:,2)-(MY);
P3=PANMS(:,:,3)-(MY);

if (d==4)
    M4=MS(:,:,4)-(MX);
    P4=PANMS(:,:,4)-(MY);
    QX= (1/d-1)*((M1.^2)+(M2.^2)+(M3.^2)+(M4.^2));
    QY= (1/d-1)*((P1.^2)+(P2.^2)+(P3.^2)+(P4.^2));
    QXY= (1/d-1)*((M1.*P1)+(M2.*P2)+(M3.*P3)+(M4.*P4));
else
    QX= (1/d-1)*((M1.^2)+(M2.^2)+(M3.^2));
    QY= (1/d-1)*((P1.^2)+(P2.^2)+(P3.^2));
    QXY= (1/d-1)*((M1.*P1)+(M2.*P2)+(M3.*P3));
end


Q =(d.*((QXY.*MX).*MY))./(((QX+QY).*((MX.^2)+(MY.^2)))+eps);
[m,n]=size(Q);
T=(1/(m*n))*(sum(sum(Q)));