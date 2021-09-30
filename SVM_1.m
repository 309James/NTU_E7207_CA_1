clear;clc;
% load data
load('data_train.mat');
load('data_test.mat');
load('label_train.mat');
disp("Data Loaded!");
X=data_train;
Y=label_train;

% construction
model = fitcsvm(X(1:260,:),Y(1:260),'BoxConstraint',4,'KernelFunction','gaussian','KernelScale',0.676*2^0.5);

% test performance
[Y1,~]=predict(model,data_train(261:330,:));
count=0;
for i=1:70
    if Y1(i)==label_train(260+i)
        count=count+1;
    end
end
accuracy=count/70;
% give the labels of test data
[Y2,~]=predict(model,data_test);