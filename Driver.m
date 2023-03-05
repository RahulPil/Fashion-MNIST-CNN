trainingData = readmatrix('train.csv');

epochs = 2;
batchSize = 32;
learningRate = 0.05;

%TODO: Gotta change stuff from other stuff to one-hot
%TODO: Test-train split???


network = CNN([ConvLayer(784, outputsize, @relu, numFilters), PoolLayer(inputsize, stride), FullyConLayer(inputsize, 10, @softmax)], learningRate, batchSize);
%TODO: replace dummy values w/ real thing in layer ctors

[lengthTrainingData, ~] = size(trainingData);

numBatches = floor(lengthTrainingData/batchSize);

for num = 1:epochs
    for i = 0:numBatches-1
        for j = 1:batchSize
            network = network.feedForward(i*batchSize+j,3:end); %TODO: change which columns are used if we change the training data to one hot?
            network = network.backwards(target, actual); %TODO: fill in w/ actual stuff
        end
        network = network.networkEndBatch();
    end
end

%TODO: now test it!