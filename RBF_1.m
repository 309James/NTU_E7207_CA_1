clc;clear;

% % % %load data files
load('data_train.mat');
load('data_test.mat');
load('label_train.mat');
disp("Data Loaded!");
% Train an RBF neural network classifier using Gaussian basis function
% data_train=transpose(data_train);
% data_test=transpose(data_test);
% label_train=transpose(label_train);
% 
% net=newrbe(data_train,label_train);
% 
% label_train2=sim(net,data_train);
% 
% label_test=sim(net,data_test);

% initial
iteration_limit=0.1;
feature_num=33;
train_num=330;
test_num=21;
center_num=280;
center=[];
delta=0.707*ones(1,center_num);
weights=[];

%randomly select some centers
for i=1:center_num
    center(i,:)= data_train(i*floor(train_num/center_num),:);
end
idx=kmeans(data_train,center_num);
class=[];
for i=1:center_num
    class{i}=[];
end
for i=1:train_num
    class{idx(i)}=[class{idx(i)};data_train(i,:)];
end
for i=1:center_num
    %         BUG-当只有一个点为一类时，求和变成了行求和
    center(i,:)=sum(class{i},1)/size(class{i},1);
end

% flag=0;
% class=[];
% iter=0;
% while(~flag)
%     iter=iter+1;
%     for i=1:center_num
%         class{i}=[];
%     end
%     for i=1:train_num
%         gap=100;
%         for j=1:center_num
%             distance=norm(data_train(i,:)-center(j,:));
%             if distance<gap
%                 belong=j;
%                 gap=distance;
%             end
%         end
%         class{belong}=[class{belong};data_train(i,:)];
%     end
%     for i=1:center_num
% %         BUG-当只有一个点为一类时，求和变成了行求和
%         new_center(i,:)=sum(class{i},1)/size(class{i},1);
%         sum1(i)=norm(new_center(i,:)-center(i,:));
%     end
%     flag=0;
%     if sum(sum1)>iteration_limit
%         flag=0;
%         center=new_center;
%     else
%         flag=1;
%     end
% end
%     disp("Center ditermined!");
Theta=[];

for i=1:train_num
    for j=1:center_num
        p=norm(data_train(i,:)-center(j,:));
        Theta(i,j)=exp(-p^2/(2*delta(j)^2));
    end
end

weights=inv(Theta'*Theta)*Theta'*label_train;
%     disp("Weight Determined!");

y_train=[];
m_sum=[];
for i=1:train_num
    m=[];
    for j=1:center_num
        m(j)=exp(-norm(data_train(i,:)-center(j,:))/(2*delta(j)^2));
        m(j)=m(j)*weights(j);
    end
    m_sum(i)=sum(m);
    if sum(m)>0
        y_train(i)=1;
    else
        y_train(i)=-1;
    end
end

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
        m1(j)=exp(-norm(data_test(i,:)-center(j,:))/(2*delta(j)^2));
        m1(j)=m1(j)*weights(j);
    end
    if sum(m1)>0
        label_test(i)=1;
    else
        label_test(i)=-1;
    end
end


