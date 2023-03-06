classdef PoolLayer < Layer

    properties
        inputSize
        outputSize
        stride
        % newly created property for calculating the sensitivity
        outputInds
        max_val_coords
    end

    methods
        function obj = PoolLayer(inputSize,stride)
            obj = obj@Layer(1,1,0);
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

        % old version
        % 
        % % only thing i modifed was creating a new property called output
        % % and setting the output on line 33 to obj,output instead of just
        % % output. Reason for this was because I needed to know which values
        % % were chosen for the max values
        % function [obj, output] = forward(obj,input)
        %     output = zeros(obj.outputSize);
        %     for row = 1:obj.outputSize(1)
        %         for col = 1: obj.outputSize(2)
        %             i = (row-1)*obj.stride;
        %             j = (col-1)*obj.stride;
        %             obj.output(row,col) = max(input(i+1:i+obj.stride,j+1:j+obj.stride));
        %         end
        %     end
        %     obj.lastInput = input;
        % end
        
        % I *think* we have to compute the gradient of the previous layer
        % then we just take only the winning elements of that
        function obj = calcSensitivity(obj,prevSensitivity,prevWeight)
%             shite = prevWeight'*prevSensitivity;
%             s = zeros(obj.outputInds);
%             for i = 1:length(obj.max_val_coords)
%                 s(obj.max_val_coords(i(1)), obj.max_val_coords(i(2))) = shite(i);
%             end
            v = reshape(prevWeight'*prevSensitivity,[13,13,8]);
                        a = zeros(26,26,8);
                        for i = 1:13
                            a(1:2:end,2*(i-1)+1:2*i,:) = repmat(v(:,i,:), [1, 2]);
                            a(2:2:end,2*(i-1)+1:2*i,:) = repmat(v(:,i,:), [1, 2]);
                        end
                        s = obj.outputInds.*a;
            obj.sensitivity = s;

%             s = obj.outputInds.*reshape(prevWeight'*prevSensitivity, size(obj.outputInds));
        end

        % old
        % %ngl this was a bitch and a half to figure out not hard just
        % %annoying as fuck for some reason. Idk
        % % i think this code should work... hopefully
        % function s = calcSensitivity(obj)
        %     % saves the coordinates of the max values in the lastInput
        %     % matrix
        %       max_val_coords = [];
        % 
        %       % similar for loops that was used in the forward function. 
        %       for row=1:obj.outputSize(1)
        %           for col=1:obj.outputSize(2)
        %               i = (row-1)*obj.stride;
        %               j = (col-1)*obj.stride;
        %               % creates a window with the proper size that we want to search
        %               % although i
        %               % realize now that we might want to do it with
        %               % respect to the outputSize(1) and outputSize(2)
        %               % rather than obj.stride
        %               % if it works then dope otherwise maybe do that
        %               % change
        %               window = obj.lastInput(i+1:i+obj.stride,j+1:j+obj.stride);
        %               % if the max value of the current stride is in the window that we are
        %               % currently looking at
        %               if any(window(:) == obj.output(row, col))
        %                   % append the coordinates (row, col) of the max
        %                   % value in the corrosponding output row and col
        %                   % using the find function. The function is going
        %                   % to return the first occurance of the value in
        %                   % case of repeat values in the window
        %                   coords = find(window == obj.output(row, col), 1, "first")
        %                   max_val_coords = [max_val_coords:coords];
        %               end
        %           end
        %       end
        %       % basically create a matrix x that has a logical index of
        %       % which elements to keep or not so that all other values can
        %       % be set to 0 except for the specific coordinates of the max
        %       % values
        %       x = false(size(obj.lastInput));
        %       x(sub2ind(size(obj.lastInput), max_val_coords(:,1), max_val_coords(:,2))) = true;
        %       % setting all values that are not true to 0
        %       s(~x)=0;                           
        % end
        % purely only here because if we dont have this and the following
        % functions then this technically isnt a subclass but an abstract
        % class so it wont be able to fall under the layer class
        function obj = updateLayer(obj,varargin)
        end
    end
end
