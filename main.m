clc;
close all;
clear all;



%%
% 读入包含叶子和背景的彩色图片
image = imread('63.png');  % 替换为你的图片路径

% 显示原始图片
figure;
imshow(image);
title('Original Image');

% 转换为灰度图像
grayImage = rgb2gray(image);

% 使用阈值分割方法（可根据实际情况调整阈值）
threshold = graythresh(grayImage);
binaryImage = imbinarize(grayImage, threshold);

% 对二值图像进行后处理（去除小的区域）
binaryImage = bwareaopen(binaryImage, 1000);

% 显示分割的二值图像
figure;
imshow(binaryImage);
title('Binary Segmentation Image');

% 创建彩色蒙版图像
colorMask = ~cat(3, binaryImage, binaryImage, binaryImage);



% 将彩色蒙版叠加到原始图片上
segmentedResult = image;
% 获取满足条件的像素索引
indices = find(colorMask);
% 将这些像素设置为白色
segmentedResult(indices) = 255;

% 显示彩色蒙版图像
figure;
imshow(segmentedResult);
title('Colored Segmentation Result');

%%
% 读入图像和真实分割
predictedSegmentation = imread('自动扣.png');
groundTruth = imread('手动扣.png');

% 将图像转换为二进制图像（如果不是的话）
predictedBinary = im2bw(predictedSegmentation);
groundTruthBinary = im2bw(groundTruth);

% 调整图像大小，使其具有相同的大小
[height, width] = size(predictedBinary);
groundTruthBinary = imresize(groundTruthBinary, [height, width]);

% 计算交集和并集
intersection = predictedBinary & groundTruthBinary;
union = predictedBinary | groundTruthBinary;

% 计算每个类别的交并比
classIoU = sum(intersection(:)) / sum(union(:));

% 计算平均交并比
numClasses = max([predictedBinary(:); groundTruthBinary(:)]);
meanIoU = sum(classIoU) / numClasses;

% 显示结果
disp(['Mean Intersection over Union (MIoU): ' num2str(meanIoU)]);
