clear;
close all;

src_dir = './';
addpath('./RAISR');
addpath('./index');
addpath('./helper');

train_img_dir = [src_dir, '../Train/'];
train_lr_dir = [src_dir,'../low_res/'];
train_hr_dir = [src_dir,'../ground_truth/'];
test_img_dir = [src_dir, '../Set14/'];
result_dir = [src_dir, '../RAISR_Set14_results/'];
images = dir(test_img_dir);
images = getFileList(images);

patch_size = 11;
scale = 2;
Q_angle = 24;  % Quantization factor for angle
Q_strenth = 3;  % Quantization factor for strength
Q_coherence = 3;  % Quantization factor for coherence
filtersname = 'filters2x_20190514_gaussian.mat';
%% Training

% comment the following two lines if it has trained.
% filters = RAISR_train(train_lr_dir,train_hr_dir, scale, patch_size, Q_angle, Q_strenth, Q_coherence,filtersname);
% save(filtersname, 'filters');
load(filtersname);


%% Testing

psnr_sum = 0;
ssim_sum = 0;
total_time = 0;

% randomly choose 3 results to show
% show_idx = randperm(length(images));
% show_idx = show_idx(1 : 3);
show_idx = [1 3];
for i =show_idx
    
    image_name = images(i).name;
    fprintf('Running test on %s...\n', image_name);
    
    image = imread([test_img_dir, image_name]);
    
    
    fprintf('Creating 1/3 LR image by bicubic interploation...\n');
    LR_image = RAISR_createLRImage(image, scale);
    LR_image = uint8(LR_image);
    
    bic_img = imresize(LR_image,scale,'bicubic');
%     imwrite(LR_image, [result_dir, 'LR_', image_name]);
    
    fprintf('Performing super resoltion to origin HR by clustering...\n');
    tic;
    
    RAISR_HR_image = RAISR(LR_image, filters, patch_size, scale, Q_angle, Q_strenth, Q_coherence);
%      figure,imshow(RAISR(image, filters, patch_size, scale, Q_angle, Q_strenth, Q_coherence));
    time = toc;
    fprintf('Time: %.2fs\n', time)
%     imwrite(RAISR_HR_image, [result_dir, 'RAISR_', image_name]);
    
    % trim to the same size as result HR
    test_HR_image = image;
    [HR_height, HR_width, c] = size(test_HR_image);
    height_trim = HR_height - mod(HR_height, scale);
    width_trim = HR_width - mod(HR_width, scale);  % not trim, 1600 -> 534, will ceil
    test_HR_image = test_HR_image(1 : height_trim, 1 : width_trim, :);

    psnr = PSNR(test_HR_image, RAISR_HR_image);
    ssim = SSIM(test_HR_image, RAISR_HR_image);
    
    psnr_bic = PSNR(test_HR_image, bic_img);
    ssim_bic = SSIM(test_HR_image, bic_img);
    
    fprintf('PSNR: %.2f\n', psnr);
    fprintf('SSIM: %.2f\n\n', ssim);
    
    psnr_sum = psnr_sum + psnr;
    ssim_sum = ssim_sum + ssim;
    total_time = total_time + time;
    
    if (ismember(i, show_idx))
        
        figure; imshow(image); title('Origin');
        figure; imshow(bic_img); title(['bicubic/PSNR: ',...
            num2str(psnr_bic, 4), 'dB, SSIM: ', num2str(ssim_bic, 2)]); 
        figure; imshow(RAISR_HR_image); title(['RAISR / PSNR: ', ...
            num2str(psnr, 4), 'dB, SSIM: ', num2str(ssim, 2)]); 
    end
    imwrite(bic_img,['bic',images(i).name]);
    imwrite(RAISR_HR_image,['raisr',images(i).name]);
    imwrite(image,'origin.bmp');
end





