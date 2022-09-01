clear all
clc
close all

pan_path = strcat(pwd,'/','../Dataset/TestFolder/pan/');
pan_label_path = strcat(pwd,'/','../Dataset/TestFolder/pan_label/');
ms_path = strcat(pwd,'/','../Dataset/TestFolder/ms/');
ms_label_path = strcat(pwd,'/','../Dataset/TestFolder/ms_label/');

lr_fusion_path = strcat(pwd,'/','../IMageFolder/FusionFolderLR/');
hr_fusion_path = strcat(pwd,'/','../IMageFolder/FusionFolderHR/');

N = 212
sensor = 'QB'
value =  Geteval(ms_label_path,lr_fusion_path,pan_label_path,ms_label_path,hr_fusion_path,N,sensor);
disp('     ergas     rmse      rase      qave      sam       ssim      fsim      qnr      D_lambda      D_s');
disp(value)
