% trainingData = readmatrix('train.csv');
% labels = uint8(trainingData(:,2))+1; % these are from 1-10!!!!
% trainingData = trainingData(:,3:end)/255;
% testData = trainingData(50001:end,:);
% trainingData = trainingData(1:50000,:);
% testLabels = labels(50001:end);

epochs = 1;
batchSize = 100;
decayRate = 0.002;
learningRate = 0.01;
conv1 = ConvLayer([28 28],[26,26,9],[3,3,1], @relu);
pool1 = PoolLayer(conv1.outputSize, 2);
conv2 = ConvLayer(pool1.outputSize, [11,11,17],[3,3,9], @relu);
conv3 = ConvLayer(conv2.outputSize,[9,9,33],[3,3,17],@relu);
pool2 = PoolLayer(conv3.outputSize,2);
fullycon1 = FullyConLayer2(5*5*33, 35, @relu);
fullycon2 = FullyConLayer(35, 10, @softmax);
layers = {conv1 pool1 conv2 conv3 pool2 fullycon1 fullycon2};
          
network = CNN(layers, learningRate, batchSize,decayRate);


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
        fprintf("epoch %d batch %d\n",num,i);
        loss = 0;
        for j = 1:batchSize
            it = i*batchSize+j;
            [network,actual] = network.feedForward(reshape(trainingData(it,:),28,28));
            loss = loss - log(actual(labels(it)));   % cross entropy loss
            network = network.backwards(target(:,labels(it)), actual);
        end
        loss = loss/batchSize;
        % if loss<bestloss
        %     bestnet = copy(network);
        %     bestloss = loss;
        % end

        network = network.networkEndBatch((num-1)*numBatches+i+1);

        addpoints(h, [(num-1)*numBatches+i+1], [loss]);
        drawnow limitrate
    end
end

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

