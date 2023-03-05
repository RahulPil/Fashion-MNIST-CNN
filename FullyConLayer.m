classdef FullyConLayer < Layer
    %UNTITLED5 Summary of this class goes here
    %   Detailed explanation goes here

    methods
        function obj = FullyConLayer(inputSize, outputSize, transfer)
           obj = obj@Layer(inputSize,outputSize,transfer);
        end

        function [obj, output] = forward(obj,input)
            output = obj.transfer(obj.weightMatrix*input+obj.biasVector);
            obj.lastInput = input;
        end

        function obj = calcSensitivity(obj, prevOut, nextSens, nextWeight) % do we need to store net input?
        end

        function obj = calcLastSensitivity(obj, error, output)
        end

        function obj = updateLayer(obj, sensitivity, prevOut)
        end
    end
end