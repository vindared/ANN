%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Vorlesung: Maschinelles Lernen in der Regelungstechnik    %
% Lehrstuhl für Regelungstechnik                            %
% Tutor: Hartwig Huber, hartwig.huber@fau.de                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function dJ = derivedCostFunction(obj,yReal,yNet)
%DERIVAEDCOSTFUNCTION calculates the derivative of the cost function 
% i.e. dJ = dJ/dyNet
    
    
    if ~isvector(yReal)
        error('yReal must be a vector or scalar');
    end
    if ~isvector(yNet)
        error('yNet must be a vector or scalar');
    end
    
    yReal = yReal(:);
    yNet = yNet(:);
    
    % cross entropy:
    %% Hier erweitern! Aufgabe c)
    dJ = 0;
    
end

