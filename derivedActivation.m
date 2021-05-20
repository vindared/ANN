%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Vorlesung: Maschinelles Lernen in der Regelungstechnik    %
% Lehrstuhl für Regelungstechnik                            %
% Tutor: Hartwig Huber, hartwig.huber@fau.de                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dz = derivedActivation(obj,a,output)
%DERIVEDACTIVATION derivative of activation function. Same as activation.m
% but returns the derivative.
    
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
            dz = 0;
        case 'tanh'
            % tanh
            %% Hier erweitern! Aufgabe c)
            dz = 0;
        case 'lin'
            % lin
            %% Hier erweitern! Aufgabe c)
            dz = 0;
        case 'relu'
            % ReLU
            %% Hier erweitern! Aufgabe c)
            dz = 0;
        otherwise
            error('activation function type (eighter hType or sigmaType) is invalid!');
    end
    
end

