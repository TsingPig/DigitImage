clc;
close all;
clear all;

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
