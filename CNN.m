classdef CNN
    properties
        layers
        learningRate

        batchIncrementor
    end

    methods 
        function obj = CNN(layersList,theLearningRate)
            obj.layers = layersList;
            obj.learningRate = theLearningRate;
        end

        function [obj, output] = feedForward(obj, input)
            [obj, temp] = obj.layers(1).forward(input);
            for i = 2:length(obj.layers)
                [obj, temp] = obj.layers(i).forward(temp);
            end
            output = temp;
        end

        function obj = backwards(obj, target, actual)
            obj = obj.layers(end).calcLastSensitivity(target-actual);
            for i = length(obj.layers)-1:-1:1
                obj = obj.layers(i).calcSensitivity(obj.layers(i+1).sensivity, obj.layers(i+1).weightMatrix);
            end
        end

        function obj = networkEndBatch(obj)
            for i = 1:length(obj.layers)
                obj.layers(i).endBatch(obj.batchIncrementor);
            end
            obj.batchIncrementor = 0;
        end
    end
end