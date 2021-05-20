%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Vorlesung: Maschinelles Lernen in der Regelungstechnik    %
% Lehrstuhl für Regelungstechnik                            %
% Tutor: Hartwig Huber, hartwig.huber@fau.de                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function backpropagate(obj,yReal)
%BACKPROPAGATE calculates the errors in the network using backpropagation.
% Uses the activations, weights and outputs currently available in ANN obj. 
    
    if ~isvector(yReal)
        error('yReal must be a vector or scalar');
    end
    yReal = yReal(:);

    % calculate error in output layer dJ/da
    obj.delta(obj.nL+1).L(:,1) = obj.derivedCostFunction(yReal, obj.z(obj.nL+1).L) * obj.derivedActivation(obj.a(obj.nL+1).L,1);
    
    % loop over hidden layers
    for l = obj.nL:-1:1
        % loop over neurons
        for j = 1:obj.nNeurons(l)
            % sum up delta*w
            tmp = 0;
            if (l == obj.nL)
                for k = 1:obj.ny
                    tmp = tmp + obj.delta(l+1).L(k)*obj.w(l+1).L(j,k);
                end
            else
                for k = 1:obj.nNeurons(l+1)
                    tmp = tmp + obj.delta(l+1).L(k)*obj.w(l+1).L(j,k);
                end
            end
            % calculate error
            obj.delta(l).L(j,1) = obj.derivedActivation(obj.a(l).L(j),1)*tmp;
        end
    end

    
end

