classdef ConvLayer < Layer
% A convolutional layer
%   Inherits from the Layer base class

    properties
        outputSize
    end

    methods (Static)
        function saymyname()
            disp("conv");
        end
    end

    methods
        function obj = ConvLayer(inputSize,outputSize,filterSize,transfer)
            obj = obj@Layer(filterSize(1)*filterSize(2)*filterSize(3),outputSize(3),transfer);
            obj.weightMatrix = reshape(obj.weightMatrix,[filterSize outputSize(3)]);
            obj.outputSize = outputSize;
            obj.biasVector = permute(repmat(obj.biasVector,[1,inputSize(2)-filterSize(1)+1,inputSize(1)-filterSize(2)+1]),[3 2 1]);
            obj.batchNewBiases = zeros(size(obj.biasVector));
            obj.batchNewWeights = zeros(size(obj.weightMatrix));
        end

        function [obj, output] = forward(obj, input)
            obj.lastInput = input;
            temp = zeros(obj.outputSize);
            for i=1:obj.outputSize(3)
                temp(:,:,i) = convn(obj.lastInput,obj.weightMatrix(:,:,:,i),'valid');
            end
            output = obj.transferFunc(temp+obj.biasVector);
        end

        function [obj, s] = calcSensitivity(obj, prevSensitivity,~)
            % this is the one where we have a 180 flipped kernel that
            % convolves over the gradient recieved from the maxpool layer
            r = rot90(obj.weightMatrix, 2);
            s = zeros(size(obj.lastInput));
            for i=1:size(obj.lastInput,3)
                c = convn(prevSensitivity, r(:,:,:,i), 'full');
                s(:,:,i) = c(:,:,(length(obj.weightMatrix(3))-1)/2+1);
            end
            obj.sensitivity = s;
        end

        function obj = updateLayer(obj, prevSensitivity)
            % add gradient dLoss/dFilter to batch gradient
            for i=1:obj.outputSize(3)
                obj.batchNewWeights(:,:,:,i) = obj.batchNewWeights(:,:,:,i) + convn(obj.lastInput,prevSensitivity(:,:,i),'valid');
            end

            % turns out the gradient of the bias is just the sum of the
            % next layers' gradient
            b = repmat(sum(prevSensitivity,[1,2]),[size(obj.biasVector,1),size(obj.biasVector,2),1]);

            obj.batchNewBiases = obj.batchNewBiases+b;
        end
    end
    methods (Access=protected)
        function cp = copyElement(obj)
            s = size(obj.weightMatrix);
            cp = ConvLayer(size(obj.lastInput),obj.outputSize,s(1:end-1),obj.transferFunc);
            cp.weightMatrix = obj.weightMatrix;
            cp.biasVector = obj.biasVector;
        end
    end
end
