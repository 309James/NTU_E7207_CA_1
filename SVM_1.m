clear;clc;
% load data
load('data_train.mat');
load('data_test.mat');
load('label_train.mat');
disp("Data Loaded!");
dt=[];
lt=[];
indexi=randperm(330);
for i=1:330
    dt(i,:)=data_train(indexi(i),:);
    lt(i,:)=label_train(indexi(i),:);
end
data_train=dt;
label_train=lt;

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
[label_test,~]=predict(model,data_test);