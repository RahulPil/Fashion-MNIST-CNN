classdef FullyConLayer2 < Layer
% A version of the fully connected layer, customized to not be the final layer in the network
%   Inherits from the Layer base class

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
        
        % this is run after each input, adding the change for the current iteration to the 
        % summed changes to the weights and biases
        function obj = updateLayer(obj, prevSensitivity)
            obj.batchNewWeights = obj.batchNewWeights + obj.sensitivity*obj.lastInput';
            obj.batchNewBiases = obj.batchNewBiases + obj.sensitivity;
        end

        function obj = calcSensitivity(obj, prevWeight, prevSensitivity) 
            obj.sensitivity = diag(obj.direlu(obj.netInput))*(prevSensitivity'*prevWeight);
        end
    end

    methods (Access=private)
        % derivative of rectified linear unit (relu)
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


