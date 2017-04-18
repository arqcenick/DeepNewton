

 
dataDir = 'Set1';%fullfile('data', '291');
mkdir('train_imgs');
count = 0;
f_lst = [];
f_lst = [f_lst; dir(fullfile(dataDir, '*.png'))];
f_lst = [f_lst; dir(fullfile(dataDir, '*.bmp'))];
for f_iter = 1:numel(f_lst)
%     disp(f_iter);
    f_info = f_lst(f_iter);
    if f_info.name == '.'
        continue;
    end
    
    f_path = fullfile(dataDir,f_info.name);
    img_raw = imread(f_path);
    img_raw = rgb2gray(img_raw);
    img_raw = im2double(img_raw(:,:,1));
    
    img_size = size(img_raw);
    width = img_size(2);
    height = img_size(1);
    
    
    
    img_raw = img_raw(33:end-33,33:end-33);
    
    
    
    img_size = size(img_raw);
    
    img_save = imresize(img_raw, [128,128], 'method','bicubic');
    
    
    img_name = sprintf('train_imgs/%d', count);
    count = count + 1;
    save(img_name, 'img_save');
    
    
    
            
%             patch_name = sprintf('train_class/%d_%d',count, class);
%             
%             patch = imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 90);
%             save(patch_name, 'patch');
%             patch = imrotate(img_2(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 90);
%             save(sprintf('%s_2', patch_name), 'patch');
%             patch = imrotate(img_3(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 90);
%             save(sprintf('%s_3', patch_name), 'patch');
%             patch = imrotate(img_4(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 90);
%             save(sprintf('%s_4', patch_name), 'patch');
%             
%             count = count+1;
%             
%             patch_name = sprintf('train_class/%d_%d',count, class);
%             
%             patch = fliplr(imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 0));
%             save(patch_name, 'patch');
%             patch = fliplr(imrotate(img_2(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 0));
%             save(sprintf('%s_2', patch_name), 'patch');
%             patch = fliplr(imrotate(img_3(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 0));
%             save(sprintf('%s_3', patch_name), 'patch');
%             patch = fliplr(imrotate(img_4(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 0));
%             save(sprintf('%s_4', patch_name), 'patch');
%             
%             count = count+1;
%             
%             patch_name = sprintf('train_class/%d_%d',count, class);
%             
%             patch = fliplr(imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 90));
%             save(patch_name, 'patch');
%             patch = fliplr(imrotate(img_2(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 90));
%             save(sprintf('%s_2', patch_name), 'patch');
%             patch = fliplr(imrotate(img_3(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 90));
%             save(sprintf('%s_3', patch_name), 'patch');
%             patch = fliplr(imrotate(img_4(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 90));
%             save(sprintf('%s_4', patch_name), 'patch');
%             
%             count = count+1;
            
            
            %{
            patch_name = sprintf('aug/%d',count);
            
            patch = imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 180);
            save(patch_name, 'patch');
            patch = imrotate(img_2(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 180);
            save(sprintf('%s_2', patch_name), 'patch');
            patch = imrotate(img_3(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 180);
            save(sprintf('%s_3', patch_name), 'patch');
            patch = imrotate(img_4(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 180);
            save(sprintf('%s_4', patch_name), 'patch');
            
            count = count+1;
            
            patch_name = sprintf('aug/%d',count);
            
            patch = fliplr(imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 180));
            save(patch_name, 'patch');
            patch = fliplr(imrotate(img_2(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 180));
            save(sprintf('%s_2', patch_name), 'patch');
            patch = fliplr(imrotate(img_3(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 180));
            save(sprintf('%s_3', patch_name), 'patch');
            patch = fliplr(imrotate(img_4(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 180));
            save(sprintf('%s_4', patch_name), 'patch');
            
            count = count+1;
            
            patch_name = sprintf('aug/%d',count);
            
            patch = imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 270);
            save(patch_name, 'patch');
            patch = imrotate(img_2(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 270);
            save(sprintf('%s_2', patch_name), 'patch');
            patch = imrotate(img_3(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 270);
            save(sprintf('%s_3', patch_name), 'patch');
            patch = imrotate(img_4(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 270);
            save(sprintf('%s_4', patch_name), 'patch');
            
            count = count+1;
            
            patch_name = sprintf('aug/%d',count);
            
            patch = fliplr(imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 180));
            save(patch_name, 'patch');
            patch = fliplr(imrotate(img_2(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 180));
            save(sprintf('%s_2', patch_name), 'patch');
            patch = fliplr(imrotate(img_3(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 180));
            save(sprintf('%s_3', patch_name), 'patch');
            patch = fliplr(imrotate(img_4(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 180));
            save(sprintf('%s_4', patch_name), 'patch');
            
            count = count+1;
            %}
    
    
end
