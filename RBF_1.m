clc;clear;
% % % %load data files
load('data_train.mat');
load('data_test.mat');
load('label_train.mat');
disp("Data Loaded!");

% initial
feature_num=33;
train_num=330;
test_num=21;
center_num=81;
center=[];
weights=[];
class=[];
%SOM
net=selforgmap([9 9]);
net=train(net,data_train');
idx2=net(data_train');
idx=vec2ind(idx2);

for i=1:center_num
    class{i}=[];
end
for i=1:train_num
    class{idx(i)}=[class{idx(i)};data_train(i,:)];
end
for i=1:center_num
    center(i,:)=sum(class{i},1)/size(class{i},1);
end
% design sigma
d_max=0;
for i=1:center_num
    for j=i:center_num
        d=norm(center(i,:)-center(j,:));
        d_max=max(d_max,d);
    end
end
sigma=d_max/(2*center_num)^0.5*ones(1,center_num);

Theta=[];
for i=1:train_num
    for j=1:center_num
        p=norm(data_train(i,:)-center(j,:));
        Theta(i,j)=exp(-p^2/(2*sigma(j)^2));
    end
end
Theta(:,center_num+1)=1;
weights=inv(Theta'*Theta)*Theta'*label_train;

y_train=[];
m_sum=[];
for i=1:train_num
    m=[];
    for j=1:center_num
        m(j)=exp(-norm(data_train(i,:)-center(j,:))/(2*sigma(j)^2));
        m(j)=m(j)*weights(j);
    end
    m_sum(i)=sum(m)+weights(center_num+1);
    if sum(m)>0
        y_train(i)=1;
    else
        y_train(i)=-1;
    end
end
y_train=y_train';
correct=0;
for i=1:train_num
    if label_train(i)==y_train(i)
        correct=correct+1;
    end
end
accuracy=correct/train_num;

label_test=[];
for i=1:test_num
    m1=[];
    for j=1:center_num
        m1(j)=exp(-norm(data_test(i,:)-center(j,:))/(2*sigma(j)^2));
        m1(j)=m1(j)*weights(j);
    end
    if sum(m1)>0
        label_test(i)=1;
    else
        label_test(i)=-1;
    end
end
label_test=label_test';

