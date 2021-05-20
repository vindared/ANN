%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Vorlesung: Maschinelles Lernen in der Regelungstechnik    %
% Lehrstuhl für Regelungstechnik                            %
% Tutor: Hartwig Huber, hartwig.huber@fau.de                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function initialize(obj)
% INITIALIZE allocates necessary fields for ANN
    
    % weights
    for i = 1:obj.nL+1
        if (i == 1) % first hidden layer
            w(i).L = zeros(obj.nx + 1, obj.nNeurons(i));
        elseif (i == obj.nL+1) % output layer
            w(i).L = zeros(obj.nNeurons(i-1) + 1, obj.ny);
        else
            w(i).L = zeros(obj.nNeurons(i-1) + 1, obj.nNeurons(i));
        end
    end
    obj.w = w;
    
    % activations
    for i = 1:obj.nL+1
        if (i == 1) % first hidden layer
            a(i).L = zeros(obj.nNeurons(i),1);
        elseif (i == obj.nL+1) % output layer
            a(i).L = zeros(obj.ny,1);
        else
            a(i).L = zeros(obj.nNeurons(i),1);
        end
    end
    obj.a = a;
    
    % outputs
    for i = 1:obj.nL+1
        if (i == 1) % first hidden layer
            z(i).L = zeros(obj.nNeurons(i),1);
        elseif (i == obj.nL+1) % output layer
            z(i).L = zeros(obj.ny,1);
        else
            z(i).L = zeros(obj.nNeurons(i),1);
        end
    end
    obj.z = z;
    
    % errors
    for i = 1:obj.nL+1
        if (i == 1) % first hidden layer
            delta(i).L = zeros(obj.nNeurons(i),1);
        elseif (i == obj.nL+1) % output layer
            delta(i).L = zeros(obj.ny,1);
        else
            delta(i).L = zeros(obj.nNeurons(i),1);
        end
    end
    obj.delta = delta;
end

