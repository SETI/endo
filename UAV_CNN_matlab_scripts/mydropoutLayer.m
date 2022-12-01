function layer = mydropoutLayer( varargin )
% mydropoutLayer   Dropout layer
%
%   layer = mydropoutLayer() creates a dropout layer. During training, the
%   dropout layer will randomly set input elements to zero with a
%   probability of 0.5. This can be useful to prevent overfitting.
%
%   layer = mydropoutLayer(probability) will create a dropout layer, where
%   probability is a number between 0 and 1 which specifies the probability
%   that an element will be set to zero. The default is 0.5.
%
%   layer = mydropoutLayer(probability, 'PARAM1', VAL1) specifies optional
%   parameter name/value pairs for creating the layer:
%
%       'Name'                    - A name for the layer. The default is
%                                   ''.
%
%   This custom dropout layer is modified to work during testing as well as
%   training. 
%
%   Example:
%       % Create a dropout layer which will dropout roughly 40% of the input
%       % elements.
%
%       layer = mydropoutLayer(0.4);
%
%   See also nnet.cnn.layer.dropoutLayer, imageInputLayer, reluLayer.
%
%   <a href="matlab:helpview('deeplearning','list_of_layers')">List of Deep Learning Layers</a>

%   Copyright 2015-2018 The MathWorks, Inc.
%   modified by M. Phillips 2021

% Parse the input arguments.
inputArguments = iParseInputArguments(varargin{:});

% Create an internal representation of a dropout layer.
internalLayer = nnet.internal.cnn.layer.myDropout( ...
    inputArguments.Name, ...
    inputArguments.Probability);

% Pass the internal layer to a  function to construct a user visible
% dropout layer.
layer = nnet.cnn.layer.MyDropoutLayer(internalLayer);

end

function inputArguments = iParseInputArguments(varargin)
varargin = nnet.internal.cnn.layer.util.gatherParametersToCPU(varargin);
parser = iCreateParser();
parser.parse(varargin{:});
inputArguments = iConvertToCanonicalForm(parser);
end

function p = iCreateParser()
p = inputParser;

defaultProbability = 0.5;
defaultName = '';

addOptional(p, 'Probability', defaultProbability, @iAssertValidProbability);
addParameter(p, 'Name', defaultName, @iAssertValidLayerName);
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name)
end

function iAssertValidProbability(value)
validateattributes(value, {'numeric'}, ...
    {'scalar','real','finite','>=',0,'<=',1});
end

function inputArguments = iConvertToCanonicalForm(p)
inputArguments = struct;
inputArguments.Probability = p.Results.Probability;
inputArguments.Name = char(p.Results.Name); % make sure strings get converted to char vectors
end
