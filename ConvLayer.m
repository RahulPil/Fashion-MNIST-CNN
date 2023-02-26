classdef ConvLayer < Layer
    % convolutional layer

    properties
        numFilters % size of conv layer
    end

    methods
        function obj = ConvLayer(inputSize,outputSize,transfer,numFilters)
            obj = obj@Layer(inputSize,outputSize*numFilters,transfer);
            obj.weightMatrix = reshape(obj.weightMatrix,[inputSize,outputSize,numFilters]);
            obj.biasVector = reshape(obj.biasVector,[outputSize,numFilters]);
            obj.numFilters = numFilters;
        end

        function output = forward(obj, input)
            output = zeros(size(obj.weightMatrix),obj.numFilters);
            for i =1:obj.numFilters
                output(:,:,i) = transfer(conv2(input,obj.weightMatrix(:,:,i)));
            end
        end

        % tfw conv backprop is complicated ;-;
    end
end