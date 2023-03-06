classdef CNN
    properties
        layers
        learningRate
        batchSize
        batchIncrementor
    end

    methods 
        function obj = CNN(layersList,theLearningRate,batchSize)
            obj.layers = layersList;
            obj.learningRate = theLearningRate;
            obj.batchSize = batchSize;
        end

        function [obj, output] = feedForward(obj, input)
            [obj.layers{1}, temp] = obj.layers{1}.forward(input);
            for i = 2:length(obj.layers)
                [obj.layers{1}, temp] = obj.layers{i}.forward(temp);
            end
            output = temp;
        end

        function obj = backwards(obj, target, actual)
            obj.layers{end} = obj.layers{end}.calcLastSensitivity(target-actual).updateLayer();
            for i = length(obj.layers)-1:-1:2
                obj.layers{i} = obj.layers{i}.calcSensitivity(obj.layers{i+1}.sensitivity, obj.layers{i+1}.weightMatrix);
                s = obj.layers{i+1}.sensitivity;
                obj.layers{i} = obj.layers{i}.updateLayer(s);
            end
            obj.layers{1} = obj.layers{1}.updateLayer(obj.layers{2}.sensitivity);
        end

        function obj = networkEndBatch(obj)
            for i = 1:length(obj.layers)
                obj.layers{i}.endBatch(obj.batchSize,obj.learningRate);
            end
            obj.batchIncrementor = 0;
        end
    end
end