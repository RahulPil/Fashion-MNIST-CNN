classdef ConvLayer < Layer
    % convolutional layer

    properties
        numFilters % size of conv layer
    end

    methods
        function obj = ConvLayer(inputSize,outputSize,transfer,numFilters)
            obj = obj@Layer(inputSize,outputSize*numFilters,transfer);
            obj.weightMatrix = reshape(obj.weightMatrix,[inputSize,outputSize,numFilters]);
            obj.biasVector = reshape(obj.biasVector,[outputSize,numFilters]);
            obj.numFilters = numFilters;
        end

        function [obj, output] = forward(obj, input)
            output = zeros(size(input,1)-size(obj.weightMatrix,1)+1,size(input,2)-size(obj.weightMatrix,2)+1,obj.numFilters);
            for i =1:obj.numFilters
                output(:,:,i) = transfer(conv2(input,obj.weightMatrix(:,:,i)));
            end
            obj.lastInput = input;
        end

        function [obj, s] = calcSensitivity(obj, prevSensitivity)
            % this is the one where we have a 180 flipped kernel that
            % convolves over the gradient recieved from the maxpool layer
            % but make sure that its a full convolution
            % do we also have to actually use the direvative of the relu
            % function here? because we could but... idk i thought it might
            % be redundant but it might not be if we arent dealing with any
            % max pooling layers in the network right?
            s = conv2(prevSensitivity, rot90(obj.weightMatrix, 2), 'full');
            obj.sensitivity = s;
        end
        % in hindsight I thnk that this function should do the same thing
        % as updatelayer1 but im not sure if we wanna keep this or not
        function obj = updateLayer(obj)
        end

        function obj = updateLayer1(obj, prevSensitivity, output)
            % this is the one where we update the weights by convolving the
            % sensitivity of the preivous layer over the input image and
            % then do the regular ting for updating the weights
            obj.weightMatrix = obj.weightMatrix - obj.learningRate*(conv2(output, prevSensitivity));
            
            % updating the bias vector is slightly confusing. bias vectors
            % are overated anyway who tf even cares about them 
        end
    end
end
