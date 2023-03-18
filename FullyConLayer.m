classdef FullyConLayer < Layer
% The fully connected layer, in the case that it's the final layer in the network
%   Inherits from the Layer base class

    methods (Static)
        function saymyname()
            disp("fully connected\n");
        end
    end

    methods
        function obj = FullyConLayer(inputSize, outputSize, transfer)
           obj = obj@Layer(inputSize,outputSize,transfer);
        end

        function [obj, output] = forward(obj,input)
            input = reshape(input,[],1);
            obj.lastInput = input;
            output = obj.transferFunc(obj.weightMatrix*input+obj.biasVector);
        end
        
        % I think the derivative of the loss wrt the softmax output is just
        % softmax(netInput)-target aka -error
        function obj = calcLastSensitivity(obj, error)
            obj.sensitivity = -error;
        end

        % this is run after each input, adding the change for the current iteration to the 
        % summed changes to the weights and biases        
        function obj = updateLayer(obj)
            obj.batchNewWeights = obj.batchNewWeights + obj.sensitivity*obj.lastInput';
            obj.batchNewBiases = obj.batchNewBiases + obj.sensitivity;
        end
    end

    methods (Access=protected)
        function cp = copyElement(obj)
            cp = FullyConLayer(size(obj.weightMatrix,1),size(obj,weightMatrix,1),obj.transferFunc);
            cp.weightMatrix = obj.weightMatrix;
            cp.biasVector = obj.biasVector;
        end
    
    end
end
