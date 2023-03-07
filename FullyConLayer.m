classdef FullyConLayer < Layer
    %UNTITLED5 Summary of this class goes here
    %   Detailed explanation goes here

    methods (Static)
        function saymyname()
            disp("fully connected\n");
        end
    end

    methods
        function obj = FullyConLayer(inputSize, outputSize, transfer, batchSize)
           obj = obj@Layer(inputSize,outputSize,transfer, batchSize);
        end

        function [obj, output] = forward(obj,input)
            input = reshape(input,[],1);
            obj.lastInput = input;
            output = obj.transferFunc(obj.weightMatrix*input+obj.biasVector);
        end
        
        % I think the derivative of the loss wrt the softmax output is just
        % softmax(netInput)-target aka -error
        % this is that expression that I asked him about and he didn't know
        % how to derive but there seems to be a good derivation here:
        % https://towardsdatascience.com/derivative-of-the-softmax-function
        % -and-the-categorical-cross-entropy-loss-ffceefc081d1
        function obj = calcLastSensitivity(obj, error)
            obj.sensitivity = -error;
        end

        function obj = updateLayer(obj)
            obj.batchNewWeights = obj.batchNewWeights + obj.sensitivity*obj.lastInput';
            obj.batchNewBiases = obj.batchNewBiases + obj.sensitivity;
        end

        function obj = calcSensitivity(obj)
        end
    end
end
