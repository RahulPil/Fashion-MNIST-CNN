% trainingData = readmatrix('train.csv');
% labels = uint8(trainingData(:,2))+1; % these are from 1-10!!!!
% trainingData = uint8(trainingData(:,3:end)/255);
% testData = trainingData(50001:end,:);
% trainingData = trainingData(1:50000,:);
% testLabels = labels(50001:end);

epochs = 1;
batchSize = 100;
learningRate = 0.05;
numFilters = 1;
filterSize = 3;

layers = {ConvLayer(filterSize, numFilters, @relu, [28 28]);
          PoolLayer([28-filterSize+1,28-filterSize+1,numFilters], 2);
          FullyConLayer(13*13*numFilters, 10, @softmax)};
network = CNN(layers, learningRate, batchSize);

[lengthTrainingData, ~] = size(trainingData);

numBatches = floor(lengthTrainingData/batchSize);
target = eye(10);

trainingPerf = zeros(numBatches*epochs);
for num = 1:epochs
    for i = 0:numBatches-1
        loss = 0;
        for j = 1:batchSize
            it = i*batchSize+j;
            [network,actual] = network.feedForward(reshape(trainingData(it,:),28,28));
            loss = loss - log(actual(labels(it)));   % cross entropy loss
            network = network.backwards(target(labels(it)), actual);
        end
        trainingPerf((num-1)*numBatches+i+1) = loss/batchSize;
        network = network.networkEndBatch();
    end
end
plot(trainingPerf)


% this should happen per epoch but putting it here for now
loss = 0;
for i=1:size(testData,1)
    [network,actual] = network.feedForward(reshape(testData(i,:),28,28));
    loss = loss - log(actual(testLabels(i)));
end
loss = loss/size(testData,1)

