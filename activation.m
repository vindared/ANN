%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Vorlesung: Maschinelles Lernen in der Regelungstechnik    %
% Lehrstuhl für Regelungstechnik                            %
% Tutor: Hartwig Huber, hartwig.huber@fau.de                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function z = activation(obj,a,output)
%ACTIVATION activation function
% returns the output of a neuron using the activation function type
% specified in ANN. if output is set to 0, hType is udes, otherwise
% phiType is used.
    
    if nargin > 2
        if output % output layer
            type = obj.hType;
        else % common activation
            type = obj.phiType;
        end
    else % common activation
        type = obj.hType;
    end
    
    switch type
        case 'sigmoid'
            % sigmoid
            %% Hier erweitern! Aufgabe c)
            z = 0;
        case 'tanh'
            % tanh
            %% Hier erweitern! Aufgabe c)
            z = 0;
        case 'lin'
            % lin
            %% Hier erweitern! Aufgabe c)
            z = 0;
        case 'relu'
            % ReLU
            %% Hier erweitern! Aufgabe c)
            z = 0;
        otherwise
            error('activation function type (eighter hType or phiType) is invalid!');
    end
    
end

