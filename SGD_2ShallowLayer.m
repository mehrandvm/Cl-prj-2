clear
clc
close all

% initiate variables
eta=0.00001;
x=0.005:0.005:1;
y=0.005:0.005:1;
variables=[x;y];
epoch = 250;

% initiate learn random values
amoozesh = variables(:,1:100);
amoozesh_data = ones(1, 100);
amoozesh_error = ones(1, epoch);
for i= 1:100
    amoozesh_data(1,i) = (x(1,i)*x(1,i) + y(1,i)*y(1,i)) * humps(x(1,i));
end
amoozesh_data = amoozesh_data / max(amoozesh_data(:));

% number of nueron in each layer
neuron_count = 16;

% weights and bias in first layer
weight1 = rand(neuron_count,2); 
bias1 = rand(neuron_count ,1);
bias_list1 = ones(neuron_count ,100);
for i = 1 : neuron_count  
    bias_list1 (i,:) = bias1(i,1);
end


% weights and bias in second layer
weight2 = rand(neuron_count,neuron_count); 
bias2 = rand(neuron_count ,100);
bias_list2 = ones(neuron_count ,100);
for i = 1 : neuron_count  
    bias_list2 (i,:) = bias2(i,1);
end

% weights and bias in third layer
weight3 = rand(1,neuron_count); 
bias3 = rand(1,1);
bias_list3 = ones(1,100);
bias_list3(1,:) = bias3(1,1);


% initiate validation random values
etebarsanji = variables(:,101:150);
etebarsanji_data = ones(1, 50);
etebarsanji_error = ones(1, epoch);
an = etebarsanji_data;
for i= 101:150
    etebarsanji_data(1,i-100) = (x(1,i)*x(1,i) + y(1,i)*y(1,i)) * humps(x(1,i));
end
an2 = etebarsanji_data;
etebarsanji_data = etebarsanji_data / max(etebarsanji_data(:));

% initiate epochs
for k = 1 : epoch
    for k2 = 1 : 100
    % feedfowarding all variables (BGD)  
    % a(n) = w(n) * x + b 
    temp = bias_list1(:,1:100);
    result = weight1 * amoozesh(:,k2) + temp(:,k2);
    O1 = tansig(result); 
    temp = bias_list2(:,1:100);
    O2 = tansig(weight2 * O1 + temp(:,k2));
    temp = bias_list3(:,1:100);
    O3 = weight3 * O2 + temp(:,k2);
       
    % calculating back propagation
    amoozesh_diff3 = amoozesh_data(:,k2) - O3;
    amoozesh_error(k)=sum(amoozesh_diff3*amoozesh_diff3', 2)/(2); 
    weight3 = weight3 + eta * amoozesh_diff3 * O2';
    bias3 = bias3 + eta * amoozesh_diff3 * ones(100,1);
    bias_list3(1,:) = bias3(1,1);
    amoozesh_diff2 = weight3' * amoozesh_diff3;
    delta = amoozesh_diff2.* (4*exp(-2*result)./(1+exp(-2*result)).^2);
    weight2 = weight2 + eta * delta * O1';
    bias2 = bias2 + eta * amoozesh_diff2; 
    for i = 1 : neuron_count 
        bias_list2 (i,:) = bias2(i,1);
    end
    
    amoozesh_diff1 = weight2' * amoozesh_diff2;
    delta = amoozesh_diff1.* (4*exp(-2*result)./(1+exp(-2*result)).^2);
    weight1 = weight1 + eta * delta * amoozesh(:,k2)';
    bias1 = bias1 + eta * amoozesh_diff1; 
    for i = 1 : neuron_count 
        bias_list1 (i,:) = bias1(i,1);
    end
    end
    
    % calculating validation error    
    etebarsanji_layer1 = weight1 * etebarsanji + bias_list1(:,1:50); 
    etebarsanji_layer1 = tansig(etebarsanji_layer1); 
    etebarsanji_layer2 = weight2*etebarsanji_layer1 + bias_list2(:,1:50);
    etebarsanji_layer2 = tansig(etebarsanji_layer2); 
    etebarsanji_layer3 = weight3*etebarsanji_layer2 + bias_list3(:,1:50); 
    
    etebarsanji_diff = etebarsanji_data-etebarsanji_layer3; 
    etebarsanji_diff = sum(etebarsanji_diff*etebarsanji_diff', 2)/2; 
    etebarsanji_error(k) = etebarsanji_diff / 50;    
end

% calculating test error  
khatasanji = variables(:,151:200);
khatasanji_data = ones(1, 50);
for i= 151:200
    khatasanji_data(1,i-150) = (x(1,i)*x(1,i) + y(1,i)*y(1,i)) * humps(x(1,i));
end
khatasanji_data=khatasanji_data/max(khatasanji_data(:));

khatasanji_layer1 = weight1 * khatasanji + bias_list1 (:,1:50); 
khatasanji_layer1 = tansig(khatasanji_layer1);
khatasanji_layer2 = weight2 * khatasanji_layer1 + bias_list2(:,1:50);
khatasanji_layer2 = tansig(khatasanji_layer2);
khatasanji_layer3 = weight3 * khatasanji_layer2 + bias_list3(:,1:50);
khatasanji_error = khatasanji_data - khatasanji_layer3;
khatasanji_error = sum(khatasanji_error * khatasanji_error', 2)/(2 * 50);

% drawing plot
axis = 1:1:epoch;
plot(axis,amoozesh_error,'r');
hold on
plot(axis,etebarsanji_error,'g');
hold off
title('Green line: etebarsanji error | Red line: amoozesh error ');
ylabel('diff');
xlabel('epoch');

% final error value
E = khatasanji_error
