% trainingData = readmatrix('train.csv');
% labels = uint8(trainingData(:,2))+1; % these are from 1-10!!!!
% trainingData = trainingData(:,3:end)/255;
% testData = trainingData(50001:end,:);
% trainingData = trainingData(1:50000,:);
% testLabels = labels(50001:end);

epochs = 20;
batchSize = 100;
learningRate = 0.001;
conv1 = ConvLayer([28 28],[26,26,15],[3,3,1], @relu);
pool1 = PoolLayer(conv1.outputSize, 2);
conv2 = ConvLayer(pool1.outputSize, [11,11,31],[3,3,15], @relu);
pool2 = poolLayer(conv2.outputSize,2);
conv3 = ConvLayer(pool1.outputSize, [4,4,64],[3,3,31], @relu);
fullycon1 = FullyConLayer(4*4*64, 100, @relu);
fullycon2 = FullyConLayer(100, 10, @softmax);
layers = {conv1 pool1 conv2 pool2 conv2 fullycon1 fullycon2};
          
network = CNN(layers, learningRate, batchSize);

bestnet = network;
bestloss = 10000;

[lengthTrainingData, ~] = size(trainingData);

numBatches = floor(lengthTrainingData/batchSize);
target = eye(10);

trainingPerf = zeros(numBatches*epochs,1);
for num = 1:epochs
    disp(num);
    for i = 0:numBatches-1
        disp(i);
        loss = 0;
        for j = 1:batchSize
            it = i*batchSize+j;
            [network,actual] = network.feedForward(reshape(trainingData(it,:),28,28));
            loss = loss - log(actual(labels(it)));   % cross entropy loss
            network = network.backwards(target(:,labels(it)), actual);
        end
        loss = loss/batchSize;
        if loss<bestloss
            bestnet = copy(network);
            bestloss = loss;
        end

        trainingPerf((num-1)*numBatches+i+1) = loss;
        network = network.networkEndBatch();
    end
end
plot(trainingPerf)

% this should happen per epoch but putting it here for now
loss = 0;
acc = 0;
for i=1:size(testData,1)
    [network,actual] = network.feedForward(reshape(testData(i,:),28,28));
    loss = loss - log(actual(testLabels(i)));
    [m,ind] = max(actual);
    if ind==testLabels(i)
        acc = acc+1;
    end
end
loss = loss/size(testData,1)
acc = acc/length(testLabels)

