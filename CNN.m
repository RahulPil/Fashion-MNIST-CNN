classdef CNN
    properties
        layers
        learningRate
        momentumFactor
        baseMomentumFactor
        batchSize
        prevLoss
    end

    methods 
        function obj = CNN(layersList,theLearningRate,theMomentumFactor,batchSize)
            obj.layers = layersList;
            obj.learningRate = theLearningRate;
            obj.batchSize = batchSize;
            obj.momentumFactor = theMomentumFactor;
            obj.baseMomentumFactor = theMomentumFactor;
            obj.prevLoss = 5;
        end

        function [obj, output] = feedForward(obj, input)
            [obj.layers{1}, temp] = obj.layers{1}.forward(input);
            for i = 2:length(obj.layers)
                [obj.layers{i}, temp] = obj.layers{i}.forward(temp);
            end
            output = temp;
        end

        function obj = backwards(obj, target, actual)
            obj.layers{end} = obj.layers{end}.calcLastSensitivity(target-actual).updateLayer();
            for i = length(obj.layers)-1:-1:2
                obj.layers{i} = obj.layers{i}.calcSensitivity(obj.layers{i+1}.sensitivity, obj.layers{i+1}.weightMatrix);
                s = obj.layers{i+1}.sensitivity;
                obj.layers{i} = obj.layers{i}.updateLayer(s);
            end
            obj.layers{1} = obj.layers{1}.updateLayer(obj.layers{2}.sensitivity);
        end

        function [obj, undo] = networkEndBatch(obj, loss, batchNum)
            for i = 1:length(obj.layers)
                obj.layers{i} = obj.layers{i}.endBatch(obj.learningRate, obj.momentumFactor);
            end
            undo = false;
        end

%         function [obj, undo] = networkEndBatch(obj, loss, batchNum)
%             deltaLoss = loss-obj.prevLoss;
%             deltaLoss = mean(deltaLoss);
% 
%             for i = 1:length(obj.layers)
%                 obj.layers{i} = obj.layers{i}.endBatch(obj.learningRate, obj.momentumFactor);
%             end
% 
%             if deltaLoss < 0
%                 %disp(['(whisper) loss is ', loss, " prevLoss is ", obj.prevLoss, " and the delta is ", deltaLoss]);
%                 obj.learningRate = obj.learningRate * (1 + batchNum*0.00001);
%                 obj.momentumFactor = obj.baseMomentumFactor;
%                 undo = false;
%                 obj.prevLoss = loss;
%             elseif loss/(obj.prevLoss+0.00000001) > 1.1
%                 %disp(['loss is ', loss, " prevLoss is ", obj.prevLoss, " and the delta is ", deltaLoss]);
%                 obj = obj.networkUndoUpdate;
%                 obj.learningRate = obj.learningRate * (1 - batchNum*0.00001);
%                 obj.momentumFactor = 0;
%                 undo = true;
%             else
%                 %disp(['LOSS IS ', loss, " prevLoss is ", obj.prevLoss, " and the delta is ", deltaLoss]);
%                 obj = obj.networkResetBatch();
%                 undo = false;
%                 obj.prevLoss = loss;
%             end
%         end

        function obj = networkUndoUpdate(obj)
            for i = 1:length(obj.layers)
                obj.layers{i} = obj.layers{i}.undoUpdate();
            end
        end

        function obj = networkResetBatch(obj)
            for i = 1:length(obj.layers)
                obj.layers{i} = obj.layers{i}.resetBatch();
            end
        end
    end
end