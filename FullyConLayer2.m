classdef FullyConLayer2 < Layer
    %UNTITLED5 Summary of this class goes here
    %   Detailed explanation goes here
    properties
        netInput
    end
    methods (Static)
        function saymyname()
            disp("fully connected\n");
        end
    end

    methods
        function obj = FullyConLayer2(inputSize, outputSize, transfer)
           obj = obj@Layer(inputSize,outputSize,transfer);
        end

        function [obj, output] = forward(obj,input)
            input = reshape(input,[],1);
            obj.lastInput = input;
            obj.netInput = obj.weightMatrix*input+obj.biasVector;
            output = obj.transferFunc(obj.netInput);
        end
        
        % I think the derivative of the loss wrt the softmax output is just
        % softmax(netInput)-target aka -error
        % this is that expression that I asked him about and he didn't know
        % how to derive but there seems to be a good derivation here:
        % https://towardsdatascience.com/derivative-of-the-softmax-function
        % -and-the-categorical-cross-entropy-loss-ffceefc081d1
        
        function obj = updateLayer(obj, prevSensitivity)
            obj.batchNewWeights = obj.batchNewWeights + obj.sensitivity*obj.lastInput';
            obj.batchNewBiases = obj.batchNewBiases + obj.sensitivity;
        end

        function obj = calcSensitivity(obj, prevWeight, prevSensitivity) 
            obj.sensitivity = diag(obj.direlu(obj.netInput))*(prevSensitivity'*prevWeight);
        end
    end

    methods (Access=private)
        function output = direlu(~, input)
            b = logical(input>0);
            output = double(b);
        end
    end
    methods (Access=protected)
        function cp = copyElement(obj)
            cp = FullyConLayer2(size(obj.weightMatrix,1),size(obj,weightMatrix,1),obj.transferFunc);
            cp.weightMatrix = obj.weightMatrix;
            cp.biasVector = obj.biasVector;
        end
    end

end


