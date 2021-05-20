%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Vorlesung: Maschinelles Lernen in der Regelungstechnik    %
% Lehrstuhl für Regelungstechnik                            %
% Tutor: Hartwig Huber, hartwig.huber@fau.de                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = feedForward(obj,x)
% FEEDFORWARD Calcualtes the output y for a given input x
    
    if nargin < 2 % x must be provided
        error('Input vector must be provided!');
    end
    if ~isvector(x)
        error('Input must be vector or scalar type!');
    end
    
    x = x(:); % force x = x(n,1)

    % Loop over hidden layers
    for i = 1:obj.nL+1
        
        % calculate activation
        if (i == 1) % first hidden layer uses inputs
            obj.a(i).L(:,1) = ([1,x.']*obj.w(i).L).';
        else
            obj.a(i).L(:,1) = ([1,obj.z(i-1).L.']*obj.w(i).L).';
        end
        
        % calculate outputs
        if (i == obj.nL+1) % output layer needs to use sigmaType
            obj.z(i).L(:,1) = obj.activation(obj.a(i).L,1);
        else
            obj.z(i).L(:,1) = obj.activation(obj.a(i).L,0);
        end
        
    end
    
    % set outputs
    y = obj.z(obj.nL+1).L;

end

