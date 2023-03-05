classdef PoolLayer < Layer

    properties
        inputSize
        outputSize
        stride
        lastInput
        % newly created property for calculating the sensitivity
        outputInds
    end

    methods
        function obj = PoolLayer(inputSize,stride)
            obj.inputSize = inputSize;
            obj.outputSize = [inputSize(1)/stride inputSize(2)/stride];
            obj.stride = stride;
        end
        
        % instead of storing the output we store a matrix the size of the
        % input with 1s where the winners are
        function [obj, output] = forward(obj,input)
            output = zeros(obj.outputSize);
            obj.outputInds = zeros(obj.inputSize);
            for row = 1:obj.outputSize(1)
                for col = 1: obj.outputSize(2)
                    i = (row-1)*obj.stride;
                    j = (col-1)*obj.stride;
                    window = input(i+1:i+obj.stride,j+1:j+obj.stride);
                    [colMax, colInd] = max(window);
                    [actualMax, rowInd] = max(colMax);
                    obj.outputInds((row-1)*obj.stride+rowInd,(col-1)*obj.stride+colInd) = 1;
                    output(row,col) = actualMax;
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
        
        %ngl this was a bitch and a half to figure out not hard just
        %annoying as fuck for some reason. Idk
        % i think this code should work... hopefully
        function s = calcSensitivity(obj,prevSensitivity,prevWeight)
            s = obj.outputInds.*prevWeight'*prevSensitivity;
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
        function obj = updateLayer(obj,~)
        end
    end
end
