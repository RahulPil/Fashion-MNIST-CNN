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
        function obj = ConvLayer(filterSize,numFilters,transfer,batchSize,inputSize)
            obj = obj@Layer(filterSize^2,numFilters,transfer, batchSize);
            obj.weightMatrix = reshape(obj.weightMatrix,[filterSize,filterSize,numFilters]);
            obj.numFilters = numFilters;
            obj.biasVector = permute(repmat(obj.biasVector,[1,inputSize(2)-filterSize+1,inputSize(1)-filterSize+1]),[3 2 1]);
            obj.batchNewBiases = zeros(size(obj.biasVector));
            obj.batchNewWeights = zeros(size(obj.weightMatrix));
        end

        function [obj, output] = forward(obj, input)
            obj.lastInput = input;
            temp = zeros(size(obj.lastInput)-size(obj.weightMatrix)+[1 1 obj.numFilters]);
            for i=1:obj.numFilters
                temp(:,:,i) = conv2(obj.lastInput(:,:,i),obj.weightMatrix(:,:,i),'valid');
            end
            output = obj.transferFunc(temp+obj.biasVector);
        end

        function [obj, s] = calcSensitivity(obj, prevSensitivity,~)
            % this is the one where we have a 180 flipped kernel that
            % convolves over the gradient recieved from the maxpool layer
            % but make sure that its a full convolution
            % do we also have to actually use the direvative of the relu
            % function here? because we could but... idk i thought it might
            % be redundant but it might not be if we arent dealing with any
            % max pooling layers in the network right?
            r = rot90(obj.weightMatrix, 2);
            if size(prevSensitivity,3)==1
                p = repmat(prevSensitivity,[1,1,obj.numFilters]);
            else
                p = prevSensitivity;
            end
            s = zeros(size(p)+size(obj.weightMatrix)-[1 1 obj.numFilters]);
            for i=1:obj.numFilters
                s(:,:,i) = conv2(p(:,:,i), r(:,:,i), 'full');
            end
            obj.sensitivity = s;
        end

        function obj = updateLayer(obj, prevSensitivity)
            % add gradient dLoss/dFilter to batch gradient

            for i=1:obj.numFilters
                obj.batchNewWeights(:,:,i) = obj.batchNewWeights(:,:,i) + conv2(obj.lastInput(:,:,i),prevSensitivity(:,:,i),'valid');
            end

            
            % turns out the gradient of the bias is just the sum of the
            % next layers' gradient?
            b = repmat(sum(prevSensitivity,[1,2]),[size(obj.biasVector,1),size(obj.biasVector,2),1]);
            % size(b)
            % b = reshape(b, [26, 26, obj.numFilters]);        % this line might very well be problematic
%             obj.batchNewBiases = obj.batchNewBiases+permute(repmat(sum(prevSensitivity,[1,2]),[1,size(obj.biasVector,2),size(obj.biasVector,1)]),[3 2 1]);]
            obj.batchNewBiases = obj.batchNewBiases+b;
        end
    end
end
