clear;clc;
load('data_train.mat');
load('data_test.mat');
load('label_train.mat');
disp("Data Loaded!");
X=data_train;
Y=label_train;
train_num=330;
test_num=21;

model = fitcsvm(X,Y,'BoxConstraint',10,'KernelFunction','gaussian','KernelScale',2^0.5*2);
% [Y1,~]=predict(model,data_train);
% count=0;
% for i=1:train_num
%     if Y1(i)==label_train(i)
%         count=count+1;
%     end
% end
% accuracy=count/train_num;

[Y2,~]=predict(model,data_test);