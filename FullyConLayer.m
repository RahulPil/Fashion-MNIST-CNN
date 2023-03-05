classdef FullyConLayer < Layer
    %UNTITLED5 Summary of this class goes here
    %   Detailed explanation goes here
    properties
        sensitivity
    end

    methods
        function obj = FullyConLayer(inputSize, outputSize, transfer)
           obj = obj@Layer(inputSize,outputSize,transfer);
        end

        function [obj, output] = forward(obj,input)
            output = obj.transfer(obj.weightMatrix*input+obj.biasVector);
        end
        
        %i just dont know the direvative of the softmax function but once i
        %get that i think all we have to do is simply replace the (1-a) .*a
        %with the direvative of the softmax function cause the prior is the
        %direvative of the logsigmoid function
        function obj = calcSensitivity(obj, error, a)
            s = -2*diag((1 - a) .* a) * e;
            obj.sensitivity = s;
        end

        function obj = updateLayer(obj, prevOutput)
            obj.weightMatrix = obj.weightMatrix - obj.learningRate*obj.sensitivity*prevOutput';
            obj.biasVector = obj.biasVector - obj.learningRate*obj.sensitivity;
        end
    end
end
