classdef PoolLayer < Layer

    properties
        inputSize
        outputSize
        stride
        % newly created property for calculating the sensitivity
        outputInds
        max_val_coords
    end

    methods (Static)
        function saymyname()
            disp("pool\n");
        end
    end

    methods
        function obj = PoolLayer(inputSize,stride)
            obj = obj@Layer(1,1,0,0);
            obj.inputSize = inputSize;
            obj.outputSize = [inputSize(1)/stride inputSize(2)/stride inputSize(3)];
            obj.stride = stride;
        end
        
        % instead of storing the output we store a matrix the size of the
        % input with 1s where the winners are
        function [obj, output] = forward(obj,input)
            output = zeros(obj.outputSize);
            %obj.max_val_coords = [];
            obj.outputInds = zeros(obj.inputSize);
            for channel=1:obj.outputSize(3)
                for row = 1:obj.outputSize(1)
                    for col = 1: obj.outputSize(2)
                        i = (row-1)*obj.stride;
                        j = (col-1)*obj.stride;
                        window = input(i+1:i+obj.stride,j+1:j+obj.stride,channel);
                        [colMax, colInd] = max(window);
                        [actualMax, rowInd] = max(colMax);
                        obj.outputInds((row-1)*obj.stride+rowInd,(col-1)*obj.stride+colInd,channel) = 1;
                        output(row,col,channel) = actualMax;
                        %obj.max_val_coords(end+1) = [rowInd, colInd];
                    end
                end
            end
            obj.lastInput = input;
        end
        
        % I *think* we have to compute the gradient of the previous layer
        % then we just take only the winning elements of that
        function obj = calcSensitivity(obj,prevSensitivity,prevWeight)
%             shite = prevWeight'*prevSensitivity;
%             s = zeros(obj.outputInds);
%             for i = 1:length(obj.max_val_coords)
%                 s(obj.max_val_coords(i(1)), obj.max_val_coords(i(2))) = shite(i);
%             end
            v = reshape(prevWeight'*prevSensitivity,[obj.inputSize(1)/2,obj.inputSize(2)/2,obj.inputSize(3)]);
                        a = zeros(obj.inputSize);
                        for i = 1:obj.inputSize(1)/2
                            a(1:2:end,2*(i-1)+1:2*i,:) = repmat(v(:,i,:), [1, 2]);
                            a(2:2:end,2*(i-1)+1:2*i,:) = repmat(v(:,i,:), [1, 2]);
                        end
                        s = obj.outputInds.*a;
            obj.sensitivity = s;

%             s = obj.outputInds.*reshape(prevWeight'*prevSensitivity, size(obj.outputInds));
        end

        

        % purely only here because if we dont have this and the following
        % functions then this technically isnt a subclass but an abstract
        % class so it wont be able to fall under the layer class
        function obj = updateLayer(obj,varargin)
        end
    end
end
