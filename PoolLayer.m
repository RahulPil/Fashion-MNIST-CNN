classdef PoolLayer < Layer
    %UNTITLED4 Summary of this class goes here
    %   Detailed explanation goes here

    properties
        inputSize
        outputSize
        stride
    end

    methods
        function obj = PoolLayer(inputSize,stride)
            obj.inputSize = inputSize;
            obj.outputSize = [inputSize(1)/stride inputSize(2)/stride];
            obj.stride = stride;
        end

        function output = forward(obj,input)
            output = zeros(obj.outputSize);
            for row = 1:obj.outputSize(1)
                for col = 1: obj.outputSize(0)
                    i = (row-1)*obj.stride;
                    j = (col-1)*obj.stride;
                    % need to somehow store which ones we are picking
                    output(row,col) = max(input(i+1:i+obj.stride,j+1:j+obj.stride));
                end
            end
        end

        function obj = calcSensitivity(obj, prevOut, nextSens, nextWeight) % do we need to store net input?
            % it's grad of next layer but only on the elements that were
            % max
        end

        function obj = updateLayer(obj, sensitivity, prevOut)
            % idfk
        end
    end
end