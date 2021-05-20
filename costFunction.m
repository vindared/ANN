%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Vorlesung: Maschinelles Lernen in der Regelungstechnik    %
% Lehrstuhl für Regelungstechnik                            %
% Tutor: Hartwig Huber, hartwig.huber@fau.de                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function J = costFunction(obj,yReal,yNet)
%COSTFUNCTION calculates the loss of a single data sample (not the true sum
% of losses!). yReal is the reference, yNet the output of the feedForward.
    
    if ~isvector(yReal)
        error('yReal must be a vector or scalar');
    end
    if ~isvector(yNet)
        error('yNet must be a vector or scalar');
    end
    
    yReal = yReal(:);
    yNet = yNet(:);
    
    % cross-entropy-loss:
    %% Hier erweitern! Aufgabe c)
    J = 0;
    
end

