classdef Layer < handle
    % layer abstract class

    properties
        weightMatrix
        biasVector
        sensitivity
        transfer
        learningRate
        lastInput
        netOutput

        % hold info for the current batch
        batchNewWeights
        batchNewBiases
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
        [obj, output] = forward(obj,input) 

        obj = calcSensitivity(obj, nextSens, nextWeight) % do we need to store net input? Yeah, I think so
        %should update both "batch" variables

        obj = calcLastSensitivity(obj, error)
        %should update both "batch" variables

        %obj = updateLayer(obj)

        function obj = endBatch(obj, batchSize)
            obj.weightMatrix = obj.weightMatrix - (obj.learningRate/batchSize)*obj.batchNewWeights;
            obj.biasVector = obj.biasVector - (obj.learningRate/batchSize)*obj.batchNewBiases;

            obj.batchNewWeights(:) = 0;
            obj.batchNewBiases(:) = 0;
        end
    end
end