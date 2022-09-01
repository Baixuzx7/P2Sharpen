function value = Geteval(labelpath,fusionpath,panpath,mspath,nrfusionpath,N,sensor)
    discard_list = [
    ];
    startindex = 0;
    %% ERGAS
    disp('ERGAS')
    tic
    ergas = zeros([1,N]);
    idx = 1;
    for i = startindex : 1 : N- 1
        if (~ismember(i,discard_list))
            MS = imread(strcat(labelpath ,num2str(i),'.tif'));
            Fusion = imread(strcat(fusionpath,num2str(i),'.tif'));
            ergas(idx) = ERGAS(MS,Fusion);
            idx = idx + 1;
        end
    end
    toc
    ergas = ergas(1,1 : idx-1);
    ergas_mean = mean(ergas)
    ergas_var = var(ergas)
    %% RMSE
    disp('RMSE')
    tic
    rmse = zeros([1,N]);
    idx = 1;
    for i = startindex : 1 : N- 1
        if (~ismember(i,discard_list))
            MS = imread(strcat(labelpath,num2str(i),'.tif'));
            Fusion = imread(strcat(fusionpath ,num2str(i),'.tif'));
            rmse(idx) = RMSE(MS,Fusion);
            idx = idx + 1;
        end
    end
    toc
    rmse = rmse(1,1:idx-1);
    rmse_mean = mean(rmse)
    rmse_var = var(rmse)
    %% RASE
    disp('RASE')
    tic
    rase = zeros([1,N]);
    idx = 1;
    for i = startindex : 1 : N - 1
        if (~ismember(i,discard_list))
            MS = imread(strcat(labelpath ,num2str(i),'.tif'));
            Fusion = imread(strcat(fusionpath ,num2str(i),'.tif'));
            rase(idx) = RASE(MS,Fusion);
            idx = idx + 1;
        end
    end
    toc
    rase = rase(1,1:idx-1);
    rase_mean = mean(rase)
    rase_var = var(rase)
    %% QAVE
    disp('QAVE')
    tic
    qave = zeros([1,N]);
    idx = 1;
    for i = startindex : 1 : N - 1
        if (~ismember(i,discard_list))
            MS = imread(strcat(labelpath ,num2str(i),'.tif'));
            Fusion = imread(strcat(fusionpath ,num2str(i),'.tif'));
            qave(idx) = QAVE(MS,Fusion);
            idx = idx + 1;
        end
    end
    toc
    qave = qave(1,1:idx-1);
    qave_mean = mean(qave)
    qave_var = var(qave)
    %% SAM
    disp('SAM')
    tic
    sam = zeros([1,N]);
    idx = 1;
    for i = startindex : 1 : N - 1
        if (~ismember(i,discard_list))
            MS = imread(strcat(labelpath ,num2str(i),'.tif'));
            Fusion = imread(strcat(fusionpath ,num2str(i),'.tif'));
            [SAM_index,SAM_map] = SAM(double(MS),double(Fusion));
            sam(idx) = SAM_index;
            idx = idx + 1;
        end
    end
    toc
    sam = sam(1,1:idx-1);
    sam_mean = mean(sam)
    sam_var = var(sam)
    %% SSIM
    disp('SSIM')
    tic
    ssim = zeros([1,N]);
    idx = 1;
    for i = startindex : 1 : N - 1
        if (~ismember(i,discard_list))
            MS = imread(strcat(labelpath ,num2str(i),'.tif'));
            Fusion = imread(strcat(fusionpath ,num2str(i),'.tif'));
            ssim(idx) = SSIM_4Band(MS,Fusion);
            idx = idx + 1;
        end
    end
    toc
    ssim = ssim(1,1:idx-1);
    ssim_mean = mean(ssim)
    ssim_var = var(ssim)
    %% FSIM
    disp('FSIM')
    tic
    fsim = zeros([1,N]);
    idx = 1;
    for i = startindex : 1 : N - 1 
        if (~ismember(i,discard_list))
            MS = imread(strcat(labelpath ,num2str(i),'.tif'));
            Fusion = imread(strcat(fusionpath ,num2str(i),'.tif'));
            fsim(idx) = FSIM_4Band(MS,Fusion);
            idx = idx + 1;
        end
    end
    toc
    fsim = fsim(1,1:idx-1);
    fsim_mean = mean(fsim)
    fsim_var = var(fsim)
    %% Non-Refference
    qnr = zeros([1,N]);
    dlam = zeros([1,N]);
    ds = zeros([1,N]);
    idx = 1;
     for i = startindex : 1 : N - 1 
        if (~ismember(i,discard_list))
            MS = imread(strcat(mspath ,num2str(i),'.tif'));
            Fusion = imread(strcat(nrfusionpath ,num2str(i),'.tif'));
            PAN = imread(strcat(panpath ,num2str(i),'.tif'));
            MSUP = imresize(MS,4,'bicubic');
            [QNR_index,D_lambda_index,D_s_index] = QNR(Fusion,MSUP,PAN,sensor,4);
            qnr(idx) = QNR_index;
            dlam(idx)  = D_lambda_index;
            ds(idx)  = D_s_index;
            idx = idx + 1;
        end
     end
    qnr = qnr(1,1:idx-1);
    qnr_mean = mean(qnr);
    qnr_var = var(qnr);
    dlam = dlam(1,1:idx-1);
    dlam_mean = mean(dlam);
    dlam_var = var(dlam);
    ds = ds(1,1:idx-1);
    ds_mean = mean(ds);
    ds_var = var(ds);
    
    value = [
      ergas_mean,rmse_mean,rase_mean,qave_mean,sam_mean,ssim_mean,fsim_mean,qnr_mean,dlam_mean,ds_mean;
      ergas_var,rmse_var,rase_var,qave_var,sam_var,ssim_var,fsim_var,qnr_var,dlam_var,ds_var;
    ];    
end

