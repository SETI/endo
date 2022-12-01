classdef myDropout < nnet.layer.Layer % & nnet.layer.Formattable (Optional) 

    properties
        % Dropout Layer properties.
        
        % Probability of dropout
        Probability

    end

    properties (Learnable)
        % (Optional) Layer learnable parameters.

        % Layer learnable parameters go here.
    end
    
    methods
        function layer = myDropout(name, probability)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.
            
            % Layer name
            layer.Name = name;
            
            % Layer description
            layer.Description = "Dropout layer that works during testing";
            
            % set probability
            layer.Probability = probability;

            % Layer constructor function goes here.
        end
        
        function [Z] = predict(layer, X)
            
            if ~isa(X, 'dlarray')
                superfloatOfX = superiorfloat(X);
            else
                superfloatOfX = superiorfloat(extractdata(X));
            end
            dropoutScaleFactor = cast( 1 - layer.Probability, superfloatOfX );
            dropoutMask = ( rand(size(X), 'like', X) > layer.Probability ) / dropoutScaleFactor;
            Z = X.*dropoutMask;
        end

        function [Z, dropoutMask] = forward(layer, X)
            % Use "inverted dropout", where we use scaling at training time
            % so that we don't have to scale at test time. The scaled
            % dropout mask is returned as the variable "dropoutMask".
            if ~isa(X, 'dlarray')
                superfloatOfX = superiorfloat(X);
            else
                superfloatOfX = superiorfloat(extractdata(X));
            end
            dropoutScaleFactor = cast( 1 - layer.Probability, superfloatOfX );
            dropoutMask = ( rand(size(X), 'like', X) > layer.Probability ) / dropoutScaleFactor;
            Z = X.*dropoutMask;
        end

        function [dX] = backward(~, ~, ~, dZ, mask)
            dX = dZ.*mask;
        end
    end
end