function [fsim_score]=FSIM_4Band(I_F,I_GT)

I_F_1=I_F(:,:,1);
I_F_2=I_F(:,:,2);
I_F_3=I_F(:,:,3);
I_F_4=I_F(:,:,4);
I_GT_1=I_GT(:,:,1);
I_GT_2=I_GT(:,:,2);
I_GT_3=I_GT(:,:,3);
I_GT_4=I_GT(:,:,4);

fsim_score_1=FeatureSIM(I_F_1,I_GT_1);
fsim_score_2=FeatureSIM(I_F_2,I_GT_2);
fsim_score_3=FeatureSIM(I_F_3,I_GT_3);
fsim_score_4=FeatureSIM(I_F_4,I_GT_4);
fsim_score=(fsim_score_1+fsim_score_2+fsim_score_3+fsim_score_4)/4;