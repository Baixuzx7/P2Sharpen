function sol=RMSE(M,F)
M=double(M);
F=double(F);
[n,m,d]=size(F);
D=(M(:,:,:)-F).^2;
sol=sqrt(sum(sum(sum(D)))/(n*m*d));
