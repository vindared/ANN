%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Vorlesung: Maschinelles Lernen in der Regelungstechnik    %
% Lehrstuhl für Regelungstechnik                            %
% Tutor: Hartwig Huber, hartwig.huber@fau.de                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function initWeights(obj,type)
%INITWEIGHTS inits the weights (excluding biases) with the method specified
%by type.

    if strcmp(type,'random')
        for i = 1:obj.nL+1
            if (i == 1) % first hidden layer
                obj.w(i).L(2:end,:) = randn(obj.nx, obj.nNeurons(i));
            elseif (i == obj.nL+1) % output layer
                obj.w(i).L(2:end,:) = randn(obj.nNeurons(i-1), obj.ny);
            else
                obj.w(i).L(2:end,:) = randn(obj.nNeurons(i-1), obj.nNeurons(i));
            end
        end 
    elseif strcmp(type,'zeros')
        return; % weights are already initialized to zero in 'initialize()'
    elseif strcmp(type,'he')
        for i = 1:obj.nL+1
            if (i == 1) % first hidden layer
                obj.w(i).L(2:end,:) = randn(obj.nx, obj.nNeurons(i)).*sqrt(2/(obj.nx));
            elseif (i == obj.nL+1) % output layer
                obj.w(i).L(2:end,:) = randn(obj.nNeurons(i-1), obj.ny).*sqrt(2/(obj.nNeurons(i-1)));
            else
                obj.w(i).L(2:end,:) = randn(obj.nNeurons(i-1), obj.nNeurons(i)).*sqrt(2/(obj.nNeurons(i-1)));
            end
        end 
    elseif strcmp(type,'xavier')
        for i = 1:obj.nL+1
            if (i == 1) % first hidden layer
                obj.w(i).L(2:end,:) = randn(obj.nx, obj.nNeurons(i)).*sqrt(2/(obj.nx));
            elseif (i == obj.nL+1) % output layer
                obj.w(i).L(2:end,:) = randn(obj.nNeurons(i-1), obj.ny).*sqrt(2/(obj.nNeurons(i-1)));
            else
                obj.w(i).L(2:end,:) = randn(obj.nNeurons(i-1), obj.nNeurons(i)).*sqrt(2/(obj.nNeurons(i-1)));
            end
        end 
    else
        error('No valid "type" was provided!');
    end

end

