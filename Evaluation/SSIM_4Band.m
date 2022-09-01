function [ssim_score]=SSIM_4Band(I_F,I_GT)

I_F_1=I_F(:,:,1);
I_F_2=I_F(:,:,2);
I_F_3=I_F(:,:,3);
I_F_4=I_F(:,:,4);
I_GT_1=I_GT(:,:,1);
I_GT_2=I_GT(:,:,2);
I_GT_3=I_GT(:,:,3);
I_GT_4=I_GT(:,:,4);

ssim_score_1=ssim(I_F_1,I_GT_1);
ssim_score_2=ssim(I_F_2,I_GT_2);
ssim_score_3=ssim(I_F_3,I_GT_3);
ssim_score_4=ssim(I_F_4,I_GT_4);
ssim_score=(ssim_score_1+ssim_score_2+ssim_score_3+ssim_score_4)/4;