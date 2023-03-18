classdef PoolLayer < Layer
% A max pooling layer, to simplify things after convolution has been done
%   Inherits from the base Layer class

    properties
        inputSize
        outputSize
        stride
        % newly created property for calculating the sensitivity
        outputInds
        max_val_coords
        z
    end

    methods (Static)
        function saymyname()
            disp("pool\n");
        end
    end

    methods
        function obj = PoolLayer(inputSize,stride)
            obj = obj@Layer(1,1,0);

            obj.inputSize = inputSize;
            obj.z = zeros(2*ceil(inputSize(1:2)/2));
            obj.outputSize = [ceil(inputSize(1:2)/2) inputSize(3)];
            obj.stride = stride;
            
        end
        
        % instead of storing the output we store a matrix the size of the
        % input with 1s where the winners are
        function [obj, output] = forward(obj,input)
            output = zeros(obj.outputSize);
            obj.outputInds = zeros(obj.inputSize);
            for channel=1:obj.outputSize(3)
                obj.z(1:obj.inputSize(1),1:obj.inputSize(2))=input(:,:,channel);
                for row = 1:obj.outputSize(1) % move through the input and do max pooling
                    for col = 1: obj.outputSize(2)
                        i = (row-1)*obj.stride;
                        j = (col-1)*obj.stride;
                        window = obj.z(i+1:i+obj.stride,j+1:j+obj.stride);
                        [colMax, colInd] = max(window);
                        [actualMax, rowInd] = max(colMax);
                        obj.outputInds((row-1)*obj.stride+rowInd,(col-1)*obj.stride+colInd,channel) = 1;
                        output(row,col,channel) = actualMax;
                    end
                end
            end
            obj.lastInput = input;
        end
        
        % use the "winners" and the sensitivity of the next layer to calculate this layer's sensitivities
        function obj = calcSensitivity(obj,prevSensitivity,prevWeight)
            flag=0; % store whether or not we needed to reshape

            if length(size(prevWeight))<=2 % change weights to match this layer's output
                v = reshape(prevWeight'*prevSensitivity,obj.outputSize);
            else
                v = prevSensitivity;
                flag = 1;
            end

            a = zeros([size(obj.z) obj.inputSize(3)]);
            for i = 1:size(a,1)/2
                a(1:2:end,2*(i-1)+1:2*i,:) = repmat(v(:,i,:), [1, 2]);
                a(2:2:end,2*(i-1)+1:2*i,:) = repmat(v(:,i,:), [1, 2]);
            end
            if any(size(obj.outputInds)<size(a))
                holder = zeros(size(a));
                holder(1:size(obj.outputInds,1),1:size(obj.outputInds,2),1:size(obj.outputInds,3)) = obj.outputInds;

                obj.outputInds = holder;
            end

            s = obj.outputInds.*a;
            if flag==0
                % we did reshape earlier, so now we need to go back
                obj.sensitivity = s(1:end-1,1:end-1,:);
            else
                obj.sensitivity = s;
            end
        end

        % pool layer is the one type that doesn't really need this function, 
        % so we'll leave the implementation blank
        function obj = updateLayer(obj,varargin)
        end
    end
        methods (Access=protected)
        function cp = copyElement(obj)
            cp = PoolLayer(obj.inputSize,2);
        end
    end
    
end
