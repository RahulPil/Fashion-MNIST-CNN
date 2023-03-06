
classdef Layer < handle
    properties
        weightMatrix
        biasVector
        transferFunc
        learningRate
        lastInput
        sensitivity

        %% these cannot be deleted as they are required to accumulate the
        %% change in gradient
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
        
        % too braindead to understand what the fuck this thing does but
        % sure
        % the batch training stuff is too big brained for me atm XD
        % put this here because its not a virtual method but at the same
        % time i am unaware how the actual purpose of this function and why
        % we need to carry out the batchNewWeights calculation like this.
        function obj = endBatch(obj, batchSize,learningRate)
            obj.weightMatrix = obj.weightMatrix - (learningRate/batchSize)*obj.batchNewWeights;
            obj.biasVector = obj.biasVector - (learningRate/batchSize)*obj.batchNewBiases;

            obj.batchNewWeights(:) = 0;
            obj.batchNewBiases(:) = 0;
        end
    end
    % abstract keyword is necessary for virtual methods. All methods below
    % are virtual
    methods (Abstract)
        % in matlab we dont have to specify the return variables of
        % virtual methods, they can be specified in the subclass
        % implementations
        forward(obj, varargin)
        
        % only change really made here is I added a as a parameter because
        % we technically need that to do the direvative of the transfer
        % function part of calculating the sensitivity of a generic vanilla
        % layer
        % setting the variables of all the virtual methods to only take obj
        % because we can always add these parameters into the subclass
        % implementations but if we dont actually use a parameter in the
        % superclass method then thats kinda ass. Especially cause each
        % layer does this shit differently
        % also varargin allows us to pass in extra varaibles into the
        % function implementation in the subclass wihtout the compiler
        % thinking that we are implementing a different function entirely
        calcSensitivity(obj, varargin)
        
        updateLayer(obj, varargin)

    end
end
