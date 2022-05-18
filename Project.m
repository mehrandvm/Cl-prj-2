clear
clc
close all

%  ????? ?????? X ? Y ???? ????? ???? ????
eta=0.00001;
x=0.005:0.005:1;
y=0.005:0.005:1;
variables=[x;y];
epoch = 5000;

% ????? ???? ???? ????? ???? ????
amoozesh = variables(:,1:100);
amoozesh_data = ones(1, 100);
amoozesh_error = ones(1, epoch);
for i= 1:100
    amoozesh_data(1,i) = (x(1,i)*x(1,i) + y(1,i)*y(1,i)) * humps(x(1,i));
end
amoozesh_data = amoozesh_data / max(amoozesh_data(:));

% ????? ????? ?? ?? ????
neuron_count = 16;

% ????? ?? ???? ????? ?? ??? ?? ? ????? ??? ???? ??? ??????
weight1 = rand(neuron_count,2)-0.5; 
bias1 = rand(neuron_count ,1);
% ????? ?????? ?? ????? ???? ??????? ?? ??? ???????
bias_list1 = ones(neuron_count ,100);
for i = 1 : neuron_count  
    bias_list1 (i,:) = bias1(i,1);
end

% ????? ?? ???? ????? ?? ??? ?? ? ????? ??? ???? ??? ??????
weight2 = rand(1,neuron_count )-0.5; 
bias2 = rand(1,1);
% ????? ?? ???? ????? ?? ??? ?? ? ????? ??? ???? ??? ??????
bias_list2 = ones(1,100);
bias_list2(1,:) = bias2(1,1);


% ????? ???? ???? ?????? ???? ???? ????
etebarsanji = variables(:,101:150);
etebarsanji_data = ones(1, 50);
etebarsanji_error = ones(1, epoch);
an = etebarsanji_data;
for i= 101:150
    etebarsanji_data(1,i-100) = (x(1,i)*x(1,i) + y(1,i)*y(1,i)) * humps(x(1,i));
end
an2 = etebarsanji_data;
etebarsanji_data = etebarsanji_data / max(etebarsanji_data(:));


% ????? ? ?????? ???? ???? ????
for k = 1 : epoch
    
    %     ??????? ?? ????? ??? ???? feedforward ???? ????
    % a(n) = w(n) * x + b
    %     ??????? ?? ???? ??????? ???? ???? ???? ????? ?? ????? ???? ???? ???
    result = weight1 * amoozesh + bias_list1(:,1:100);
    O1 = tansig(result); 
    O2 = weight2 * O1 + bias_list2(:,1:100);
    
    
    %     ????? ???????? backpropogation ?? ???? ??? ?? ??? ? ???? ???? ??????
    %     ???? ??????? ?? ? ????? ??
    amoozesh_diff2 = amoozesh_data - O2;
    amoozesh_error(k)=sum(amoozesh_diff2*amoozesh_diff2', 2)/(2 * 100); 
    weight2 = weight2 + eta * amoozesh_diff2 * O1';
    bias2 = bias2 + eta * amoozesh_diff2 * ones(100,1);
    bias_list2(1,:) = bias2(1,1);
    amoozesh_diff1 = weight2' * amoozesh_diff2;
    delta = amoozesh_diff1.* (4*exp(-2*result)./(1+exp(-2*result)).^2);
    weight1 = weight1 + eta * delta * amoozesh';
    bias1 = bias1 + eta * amoozesh_diff1; 
    for i = 1 : neuron_count 
        bias_list1 (i,:) = bias1(i,1);
    end
    
    
%     ?????????? ????? ?? ? ??? ??? ???? ?? ???? ?????? ???? ?????? ?? ???? ??? ?????? ????    
    etebarsanji_layer1 = weight1 * etebarsanji + bias_list1(:,1:50); 
    etebarsanji_layer1 = tansig(etebarsanji_layer1); 
    etebarsanji_layer2 = weight2*etebarsanji_layer1 + bias_list2(:,1:50); 
    
    etebarsanji_diff = etebarsanji_data-etebarsanji_layer2; 
    etebarsanji_diff = sum(etebarsanji_diff*etebarsanji_diff', 2)/2; 
    etebarsanji_error(k) = etebarsanji_diff / 50;    
end

% ????? ???? ???? ???? ??? ???? ????
khatasanji = variables(:,151:200);
khatasanji_data = ones(1, 50);
for i= 151:200
    khatasanji_data(1,i-150) = (x(1,i)*x(1,i) + y(1,i)*y(1,i)) * humps(x(1,i));
end
khatasanji_data=khatasanji_data/max(khatasanji_data(:));

% ????? ???? ???? ????
khatasanji_layer1 = weight1 * khatasanji + bias_list1 (:,1:50); 
khatasanji_layer1 = tansig(khatasanji_layer1);
khatasanji_layer2 = weight2 * khatasanji_layer1 + bias_list2(:,1:50);
khatasanji_error = khatasanji_data - khatasanji_layer2;
khatasanji_error = sum(khatasanji_error * khatasanji_error', 2)/(2 * 50);

% ??? ??????
axis = 1:1:epoch;
plot(axis,amoozesh_error,'r');
hold on
plot(axis,etebarsanji_error,'g');
hold off
title('Green line: etebarsanji error | Red line: amoozesh error ');
ylabel('diff');
xlabel('epoch');

E = khatasanji_error
