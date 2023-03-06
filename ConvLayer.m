classdef ConvLayer < Layer
    % convolutional layer

    properties
        numFilters % size of conv layer
    end

    methods (Static)
        function saymyname()
            disp("conv");
        end
    end

    methods
        function obj = ConvLayer(filterSize,numFilters,transfer,inputSize)
            obj = obj@Layer(filterSize^2,numFilters,transfer);
            obj.weightMatrix = reshape(obj.weightMatrix,[filterSize,filterSize,numFilters]);
            obj.numFilters = numFilters;
            obj.biasVector = permute(repmat(obj.biasVector,[1,inputSize(2)-filterSize+1,inputSize(1)-filterSize+1]),[3 2 1]);
            obj.batchNewBiases = zeros(size(obj.biasVector));
            obj.batchNewWeights = zeros(size(obj.weightMatrix));
        end

        function [obj, output] = forward(obj, input)
            obj.lastInput = repmat(input,[1,1,obj.numFilters]);
            output = obj.transferFunc(convn(obj.lastInput,obj.weightMatrix,'valid')+obj.biasVector);
        end

        function [obj, s] = calcSensitivity(obj, prevSensitivity,~)
            % this is the one where we have a 180 flipped kernel that
            % convolves over the gradient recieved from the maxpool layer
            % but make sure that its a full convolution
            % do we also have to actually use the direvative of the relu
            % function here? because we could but... idk i thought it might
            % be redundant but it might not be if we arent dealing with any
            % max pooling layers in the network right?
            s = convn(repmat(prevSensitivity,[1,1,obj.numFilters]), rot90(obj.weightMatrix, 2), 'full');
            obj.sensitivity = s;
        end

        function obj = updateLayer(obj, prevSensitivity)
            % add gradient dLoss/dFilter to batch gradient
            obj.batchNewWeights = obj.batchNewWeights + convn(obj.lastInput, prevSensitivity,'valid');
            
            % turns out the gradient of the bias is just the sum of the
            % next layers' gradient?
            size(sum(prevSensitivity,[1,2]))
            b = repmat(sum(prevSensitivity,[1,2]),[size(obj.biasVector,1),size(obj.biasVector,2),1]);
            % size(b)
            % b = reshape(b, [26, 26, obj.numFilters]);        % this line might very well be problematic
%             obj.batchNewBiases = obj.batchNewBiases+permute(repmat(sum(prevSensitivity,[1,2]),[1,size(obj.biasVector,2),size(obj.biasVector,1)]),[3 2 1]);]
            obj.batchNewBiases = obj.batchNewBiases+b;
        end
    end
end
