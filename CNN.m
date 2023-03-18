classdef CNN < matlab.mixin.Copyable
% The network object that the driver creates/interacts with

    properties
        layers
        learningRate
        decayRate
        batchSize
    end

    methods 
        function obj = CNN(layersList,theLearningRate,theBatchSize,theDecayRate)
            obj.layers = layersList;
            obj.learningRate = theLearningRate;
            obj.batchSize = theBatchSize;
            obj.decayRate = theDecayRate;
        end

        function [obj, output] = feedForward(obj, input)
            [obj.layers{1}, temp] = obj.layers{1}.forward(input);
            for i = 2:length(obj.layers)
                [obj.layers{i}, temp] = obj.layers{i}.forward(temp);
            end
            output = temp;
        end

        % do backpropagation
        function obj = backwards(obj, target, actual)
            obj.layers{end} = obj.layers{end}.calcLastSensitivity(target-actual).updateLayer();
            for i = length(obj.layers)-1:-1:2 % start at the end and move backwards
                obj.layers{i} = obj.layers{i}.calcSensitivity(obj.layers{i+1}.sensitivity, obj.layers{i+1}.weightMatrix);
                s = obj.layers{i+1}.sensitivity;
                obj.layers{i} = obj.layers{i}.updateLayer(s);
            end
            obj.layers{1} = obj.layers{1}.updateLayer(obj.layers{2}.sensitivity);
        end

        % at the end of each batch, command each layer to end their individual batches
        % also decay the learning rate
        function obj = networkEndBatch(obj, batchNum)
            newLearningRate = (1/(1+obj.decayRate*batchNum))*obj.learningRate;
            for i = 1:length(obj.layers)
                obj.layers{i} = obj.layers{i}.endBatch(obj.batchSize,newLearningRate);
            end
        end
    end
    methods (Access = protected)
        function cp = copyElement(obj)
            cp = CNN(0,0,0,0);
            cp.layers = arrayfun(@(x) copy(x),obj.layers);
        end
    end
end