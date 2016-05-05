%% CS294A/CS294W Softmax Exercise iris

lambda = 1e-4;
%%======================================================================
%% STEP 1: Load data
%
addpath dataset;
[m1,m2,m3,m4,class]=textread('dataset/iris.data.txt','%f %f %f %f %s','delimiter',',');
n = size (m1,1);
y = zeros(n,1);
y(strcmp(class, 'Iris-setosa')) = 1;
y(strcmp(class, 'Iris-versicolor')) = 2;
y(strcmp(class, 'Iris-virginica')) = 3;
data = [m1,m2,m3,m4,y];
randnum = randperm(n);
a = randnum(1:0.8*n);
b = randnum(0.8*n+1:n);
trainset = data(a,:);
testset = data(b,:);

traindata = trainset(:,1:4);
trainclass = trainset(:,5);
testdata = testset(:,1:4);
testclass = testset(:,5);

inputData = traindata';
inputSize = 4; % Size of input vector 
numClasses = 3;     % Number of classes 
% Randomly initialise theta
theta = 0.005 * randn(numClasses * inputSize, 1 );

%DEBUG
DEBUG = false;
if DEBUG
    inputSize = 8;
    inputData = randn(8, 100);
    labels = randi(10, 100, 1);
end
%%======================================================================
%% STEP 2: Implement softmaxCost


[cost, grad] = softmaxCosta(theta, numClasses, inputSize,lambda,inputData,trainclass);
                                     
%%======================================================================
%% STEP 3: Gradient checking
%

if DEBUG
    numGrad = computeNumericalGradient( @(x) softmaxCost(x, numClasses, ...
                                    inputSize, lambda, inputData, trainclass), theta);

    % Use this to visually compare the gradients side by side
    disp([numGrad grad]); 

    % Compare numerically computed gradients with those computed analytically
    diff = norm(numGrad-grad)/norm(numGrad+grad);
    disp(diff); 
    
end

%%======================================================================
%% STEP 4: Learning parameters

options.maxIter = 100;
softmaxModel = softmaxTrain(inputSize, numClasses, lambda, ...
                            inputData, trainclass, options);
                          

%%======================================================================
%% STEP 5: Testing
%

inputData = testdata';

[pred] = softmaxPredict(softmaxModel, inputData);

acc = mean(testclass(:) == pred(:));
fprintf('Accuracy: %0.3f%%\n', acc * 100);

