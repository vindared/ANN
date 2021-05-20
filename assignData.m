%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Vorlesung: Maschinelles Lernen in der Regelungstechnik    %
% Lehrstuhl für Regelungstechnik                            %
% Tutor: Hartwig Huber, hartwig.huber@fau.de                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function assignData(obj,X,Y)
%ASSIGNDATA Asssign training set to ANN
    
    % check data
    if (size(X,1) ~= obj.nx)
        error('X must be consistent to nx!');
    end
    if (size(Y,1) ~= obj.ny)
        error('Y must be consistent to ny!');
    end
    if(size(Y,2) ~= size(X,2))
        error('Y and X must contain the same number of data points!');
    end
    
    obj.N = size(Y,2);
    obj.X = X;
    obj.Y = Y;
end

