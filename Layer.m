classdef Layer < matlab.mixin.Copyable
% The abstract base class that the specific types of layers will inherit from
    properties
        weightMatrix
        biasVector
        transferFunc
        learningRate
        lastInput
        sensitivity

        % hold the cummulative changes for this batch
        batchNewWeights
        batchNewBiases
    end
    methods
        function obj = Layer(inputSize, outputSize,transfer)
            % initializes the weight matrix using Kaiming He initialization 
            % as seen in https://arxiv.org/pdf/1502.01852.pdf
            % (zero-mean normal dist with stdev of sqrt(2/inputSize))
            obj.transferFunc = transfer;
            obj.weightMatrix = normrnd(0,sqrt(2/inputSize),outputSize,inputSize);
            obj.batchNewWeights = zeros(size(obj.weightMatrix));
            obj.biasVector = zeros(outputSize,1);
            obj.batchNewBiases = zeros(size(obj.biasVector));
        end
        
        % at the end of a batch, apply the summed changes
        function obj = endBatch(obj, batchSize,learningRate)
            obj.weightMatrix = obj.weightMatrix - (learningRate/batchSize)*obj.batchNewWeights;
            obj.biasVector = obj.biasVector - (learningRate/batchSize)*obj.batchNewBiases;

            obj.batchNewWeights(:) = 0;
            obj.batchNewBiases(:) = 0;
        end

        
    end
    
    methods (Abstract,Access=protected)
        copyElement(obj)
    end
    % abstract keyword is necessary for virtual methods. All methods below
    % are virtual
    methods (Abstract)
        
        forward(obj, varargin)
        
        calcSensitivity(obj, varargin)
        
        updateLayer(obj, varargin)

        saymyname()
    end
end
