%% Softmax Exercise  wine
%%======================================================================
%% STEP 1 : load data
addpath dataset;

lambda = 1e-4;%initialize lambda

data=load('dataset/wine.data.txt','delimiter',',');%load data ,','separator
% data partition
n = size (data,1);
randnum = randperm(n);
a = randnum(1:0.8*n);
b = randnum(0.8*n+1:n);
trainset = data(a,:);
testset = data(b,:);

traindata = trainset(:,2:14);
trainclass = trainset(:,1);
testdata = testset(:,2:14);
testclass = testset(:,1);

inputData = traindata';
inputSize = 13; % Size of input vector 
numClasses = 3;     % Number of classes 
% Randomly initialise theta
theta = 0.005 * randn(numClasses * inputSize, 1 );


%%======================================================================
%% STEP 2: Implement softmaxCost

[cost, grad] = softmaxCost(theta, numClasses, inputSize,lambda,inputData,trainclass);
                                     
%%======================================================================
%% STEP 3: Learning parameters

options.maxIter = 100;
softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
                            inputData, trainclass, options);
                          
%%======================================================================
%% STEP 4: Testing

inputData = testdata';

[pred] = softmaxPredict(softmaxModel, inputData);

acc = mean(testclass(:) == pred(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);
