clear; clc; close all;
% img_load_test = imread('night_stars_space512bw.jpg');
% img_load_test = im2double(img_load_test);
%img_load = imread('Black_and_white_zebra.jpeg');
%img_load = im2double(img_load);


testo = load('img_original_test.mat');
a = cell2mat(struct2cell(testo));
img_original_test = reshape(a,[1000,28,28]);

testo = load('img_blur_test.mat');
a = cell2mat(struct2cell(testo));
img_blur_test = reshape(a,[1000,28,28]);

%testo3 = squeeze(b(1,:,:))';
lambda_set = logspace(-14,1,16);
MSPE_loop = zeros(1,length(lambda_set));
%lambda = 1E-4;
for kk = 1:length(lambda_set)
    lambda = lambda_set(kk);
    eyeye = eye(size(img_blur_test,2));
    img_original_compile = zeros(1000,28,28);
    img_blur_compile = zeros(1000,28,28);
    img_recover_compile = zeros(1000,28,28);
    for pp = 1:1000
        img_original_slice = squeeze(img_original_test(pp,:,:))';
        img_original_compile(pp,:,:) = img_original_slice ;

        img_blur_slice = squeeze(img_blur_test(pp,:,:))';
        img_blur_compile(pp,:,:) = img_blur_slice;

        weight_recover = inv(img_blur_slice'*img_blur_slice + lambda.*eyeye)*img_blur_slice'*img_original_slice;
        img_recover_slice = img_blur_slice*weight_recover;
        img_recover_compile(pp,:,:) = img_recover_slice;
        %pp
    end
    
    if kk == 1
        figure(1)
        imshow(squeeze(img_original_compile(1,:,:)))
        title('Original Image');

        figure(2)
        imshow(squeeze(img_blur_compile(1,:,:)))
        title('Blurred Image');

        figure(3)
        imshow(squeeze(img_recover_compile(1,:,:)))
        title('Recovered Image');
    end
    
%     figure(1)
%     imshow(squeeze(img_original_compile(end,:,:)))
%     title('Original Image');
% 
%     figure(2)
%     imshow(squeeze(img_blur_compile(end,:,:)))
%     title('Blurred Image');
% 
%     figure(3)
%     imshow(squeeze(img_recover_compile(end,:,:)))
%     title('Recovered Image');

    MSPE = sum(abs(img_original_compile-img_recover_compile).^2,'all')/numel(img_original_compile);

MSPE_loop(kk) = MSPE;
kk
end




figure(4)
semilogx(lambda_set,MSPE_loop,'color', 'k','LineWidth',2)
xlabel('Tuning parameter, \lambda')
ylabel('MSPE')
ax = gca;
ax.FontSize = 16; 
%xlabel('d_1 [nm]','FontSize', 16)
%ylabel('d_2 [nm]','FontSize', 16)




return


[r,c] = size(a);
nlay  = 3;
out   = permute(reshape(a',[c,r/nlay,nlay]),[1,2,3]);

return

img_load = table2array(readtable('MNIST_example_clear.csv'));
img_load_blur = table2array(readtable('MNIST_example_blur.csv'));
%T = readtable('MNIST_example_blur.csv')
blurred2 = img_load_blur;

figure(1)
imshow(img_load)
title('Original Image');

% figure(11)
% imshow(img_load_test)
% title('Original Test Image');

screen_width_in_pixels = max(size(img_load));
screen_width_in_mm = 25;
blur_radius_in_mm = .5;
% blur_radius_in_pixel = blur_radius_in_mm*screen_width_in_pixels/screen_width_in_mm;
% 
% H = fspecial('disk',blur_radius_in_pixel);
% blurred2 = conv2(img_load,H,'same');

blurred2_norm = mat2gray(blurred2,[0 .5]);
% blurred2_test = conv2(img_load_test,H,'same');
% blurred2_norm_test = mat2gray(blurred2_test,[0 .5]);

figure(2)
imshow(blurred2)
title('Blurred, Noisy Image');

% figure(3)
% imshow(blurred2_norm_test)
% title('Blurred, Noisy Test Image');

%% Denoising %%

%% L2 - loss cost function
lambda = 1E-4;
eyeye = eye(size(img_load,2));

X_LS2 = inv(blurred2'*blurred2 + lambda.*eyeye)*blurred2'*img_load;
%figure(5)
%imshow(X_LS2)

L2_denoise = blurred2*X_LS2;
figure(4)
imshow(L2_denoise)
title('De-Noised Image');

% L2_denoise_test = blurred2_test*X_LS2;
% figure(5)
% imshow(L2_denoise_test)
% title('De-Noised Test Image');

% wnr5 = deconvwnr(blurred2, H, 4000);
% wnr5_norm = mat2gray(wnr5,[0 .2]);
% figure(3)
% imshow(wnr5_norm)
% title('Restoration of Blurred, Noisy Image Using Estimated NSR');

% img_load2 = img_load + .001;
% blurred3 = conv2(img_load2,H);
% wnr6 = deconvwnr(blurred3, H);
% figure(13)
% imshow(wnr6)
% title('Restoration of Blurred, Noisy Image Using Estimated NSR');
% 
