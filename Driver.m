trainingData = readmatrix('train.csv');
labels = uint8(trainingData(:,2))+1; % these are from 1-10!!!!
trainingData = trainingData(:,3:end)/255;
testData = trainingData(50001:end,:);
trainingData = trainingData(1:50000,:);
testLabels = labels(50001:end);

epochs = 2;
batchSize = 100;
learningRate = 0.05;
decayRate = 0.002; %a rate of 0 gives no decay
numFilters = 10;
filterSize = 3;

layers = {ConvLayer(filterSize, numFilters, @relu, [28 28])...
          ConvLayer(filterSize, numFilters, @relu, [26 26])...
          PoolLayer([26-filterSize+1,26-filterSize+1,numFilters], 2)...
          FullyConLayer2(12*12*numFilters, 50, @relu)...
          FullyConLayer(50, 10, @softmax)};
network = CNN(layers, learningRate, batchSize, decayRate);

bestnet = network;
bestloss = 10000;

[lengthTrainingData, ~] = size(trainingData);

numBatches = floor(lengthTrainingData/batchSize);
target = eye(10);

h = animatedline;
axis([0, epochs*numBatches+10, 0, inf]);
xlabel('Batches');
ylabel('Loss');

for num = 1:epochs
    for i = 0:numBatches-1
        loss = 0;
        for j = 1:batchSize
            it = i*batchSize+j;
            [network,actual] = network.feedForward(repmat(reshape(trainingData(it,:),28,28),[1,1,numFilters]));
            loss = loss - log(actual(labels(it)));   % cross entropy loss
            network = network.backwards(target(:,labels(it)), actual);
        end
        loss = loss/batchSize;
        if loss<bestloss
            bestnet = network;
            bestloss = loss;
        end

        network = network.networkEndBatch((num-1)*numBatches+i+1);

        addpoints(h, [(num-1)*numBatches+i+1], [loss]);
        drawnow limitrate
    end
end

% this should happen per epoch but putting it here for now
loss = 0;
acc = 0;
for i=1:size(testData,1)
    [network,actual] = network.feedForward(repmat(reshape(testData(i,:),28,28),[1,1,numFilters]));
    loss = loss - log(actual(testLabels(i)));
    [m,ind] = max(actual);
    if ind==testLabels(i)
        acc = acc+1;
    end
end
loss = loss/size(testData,1)
acc = acc/length(testLabels)

