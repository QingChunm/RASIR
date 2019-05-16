function filters = RAISR_train(train_lr_img_dir,train_hr_img_dir, scale, patch_size, Q_angle, Q_strenth, Q_coherence,filtersname)
%RAISR_TRAIN train the RAISR filters
    
    half_patch_size = floor(patch_size / 2);
    
    train_lr_list = dir(train_lr_img_dir);
    train_lr_list = getFileList(train_lr_list);
    train_lr_images_num = length(train_lr_list);
    train_lr_images_path = cell(train_lr_images_num, 1);    
    
    train_hr_list = dir(train_hr_img_dir);
    train_hr_list = getFileList(train_hr_list);
    train_hr_images_num = length(train_hr_list);
    train_hr_images_path = cell(train_hr_images_num, 1);    
    
    
    hash_index_num = Q_angle * Q_strenth * Q_coherence;
    
    % record the ATA result
    Q = zeros(patch_size ^ 2, patch_size ^ 2, scale ^ 2, hash_index_num);
    
    % record the ATb result
    V = zeros(patch_size ^ 2, scale ^ 2, hash_index_num);
    cout = zeros(scale ^ 2, hash_index_num);
    for k = 1 : train_lr_images_num
        fprintf('\nExtracting patches from %d/%d...', k, train_lr_images_num);
        tic;
        train_lr_image_name = train_lr_list(k).name;
        train_lr_images_path{k} = [train_lr_img_dir, train_lr_image_name];
        lr_image = imread(train_lr_images_path{k});
        
        train_hr_image_name = train_hr_list(k).name;
        train_hr_images_path{k} = [train_hr_img_dir, train_hr_image_name];
        hr_image = imread(train_hr_images_path{k});       
        
        lr_image = rgb2gray(im2double(lr_image));
        hr_image = rgb2gray(im2double(hr_image));
        
        [height, width] = size(lr_image);
        height = scale*height;
        width = scale*width;

        bilinear_HR_image = imresize(lr_image, [height, width], 'bilinear');
        
        % padding
        bicubic_HR_image_ext = wextend('2d', 'sym', bilinear_HR_image, half_patch_size);
        
        for x = 1:1: height
            for y = 1:1 : width

                origin_LR_patch = ...
                    bicubic_HR_image_ext(x : x + 2 * half_patch_size, ...
                                         y : y + 2 * half_patch_size);
                        
                for i = 1 : 4
                    % rotate 90, filp up down and filp center
                    LR_patch = RAISR_transform(origin_LR_patch, i);
                
                    idx = RAISR_hashFunction(LR_patch, Q_angle, Q_strenth, Q_coherence);
                    type = RAISR_computeType(x, y, scale);
                    
                    LR_patch = LR_patch(:)';
                    
                    % accumulate
                    Q(:, :, type, idx) = Q(:, :, type,idx) + LR_patch' * LR_patch;
                    V(:, type, idx) = V(:, type, idx) + LR_patch' * hr_image(x, y);
                    cout(type,idx) = cout(type,idx)+1;
                end
            end
        end
        time = toc;
        fprintf('Time: %.2fs\n', time)
    end

    filters = zeros(patch_size ^ 2, scale ^ 2, hash_index_num);
    save('Q.mat', 'Q');
    save('V.mat', 'V');
    save('cout.mat', 'cout');
    % may have rank defiency warning, ignore
    warning('off');
    for type = 1 : scale ^ 2
        for idx = 1 : hash_index_num
            % solve the regression approximately
            filters(:, type, idx) = Q(:, :, type, idx) \ V(:, type, idx);
        end
    end

end

