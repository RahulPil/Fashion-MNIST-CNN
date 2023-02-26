classdef Layer < handle
    % layer abstract class

    properties
        weightMatrix
        biasVector
        sensitivity
        transfer
        learningRate
    end

    methods
        function obj = Layer(inputSize, outputSize,transfer)
            % initializes the weight matrix using Kaiming He initialization 
            % as seen in https://arxiv.org/pdf/1502.01852.pdf
            % (zero-mean normal dist with stdev of sqrt(2/inputSize))
            obj.transfer = transfer;
            obj.weightMatrix = normrnd(0,sqrt(2/inputSize),outputSize,inputSize);
            obj.biasVector = zeros(outputSize,1);
        end


        % fully virtual or some shit idfk
        function output = forward(obj,input) 
        end

        function obj = calcSensitivity(obj, prevOut, nextSens, nextWeight) % do we need to store net input?
        end

        function obj = calcLastSensitivity(obj, error, output)
        end

        function obj = updateLayer(obj, sensitivity, prevOut)
        end
    end
end