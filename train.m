%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Vorlesung: Maschinelles Lernen in der Regelungstechnik    %
% Lehrstuhl für Regelungstechnik                            %
% Tutor: Hartwig Huber, hartwig.huber@fau.de                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function train(obj)
%TRAIN uses the current data to train the ANN
    
    % reset stuff
    nB = floor(obj.N/obj.batchSize); % number of batches
    obj.J = zeros(obj.epochs*nB,1); % reset cost progression
    
    % reset weight gradients
    for i = 1:obj.nL+1
        if (i == 1) % first hidden layer
            dw(i).L = zeros(obj.nx + 1, obj.nNeurons(i));
        elseif (i == obj.nL+1) % output layer
            dw(i).L = zeros(obj.nNeurons(i-1) + 1, obj.ny);
        else
            dw(i).L = zeros(obj.nNeurons(i-1) + 1, obj.nNeurons(i));
        end
    end
    
    helpingLayer = [obj.nNeurons,obj.ny];
    
    
    % loop over epochs
    for ie = 1:obj.epochs
        
        % adjust eta in terms of epoch
        if (obj.adjustEta)
            eta = obj.eta*0.98^(1+ie);
        else
            eta = obj.eta;
        end
        
        % loop over batches
        for ib = 1:nB
            
            % loop over data
            for i = 1:obj.batchSize
                % get next sample
                xdata(:,1) = obj.X(:,(ib-1) * obj.batchSize + i);
                ydata(:,1) = obj.Y(:,(ib-1) * obj.batchSize + i);

                % feed forward
                yNet = obj.feedForward(xdata);

                % backpropagation
                obj.backpropagate(ydata);

                % calc cost average (over batch), using recursive average
                obj.J((ie-1)*nB + ib,1) = ((i-1)/i) * obj.J((ie-1)*nB + ib,1) + (1/i)*obj.costFunction(ydata,yNet);
                
                % calculate averaged gradients (recursively)
                if (ib == 1)
                    % loop over layers
                    for l = 1:obj.nL+1
                        
                        % loop over neurons
                        for j = 1:helpingLayer(l)
                            if (l == 1)
                                dw(l).L(:,j) = obj.delta(l).L(j)*[1;xdata];
                            else
                                dw(l).L(:,j) = obj.delta(l).L(j)*[1;obj.z(l-1).L];
                            end
                        end
                        
                    end
                else
                    % loop over layers
                    for l = 1:obj.nL+1
                        
                        % loop over neurons
                        for j = 1:helpingLayer(l)
                            if (l == 1)
                                dw(l).L(:,j) = ((i-1)/i) * dw(l).L(:,j) + (1/i)*obj.delta(l).L(j)*[1;xdata];
                            else
                                dw(l).L(:,j) = ((i-1)/i) * dw(l).L(:,j) + (1/i)*obj.delta(l).L(j)*[1;obj.z(l-1).L];
                            end
                        end
                        
                    end
                end
            end % end loop over data in batch
            
            % optional: clip gradients
            if obj.clipLoss
                % loop over layers
                for l = 1:obj.nL+1
                    % loop over neurons
                    for j = 1:helpingLayer(l)
                        for k = 1:size(obj.w(l).L,1)
                            if abs(dw(l).L(k,j)) > obj.maxGradient
                                dw(l).L(k,j) = sign(dw(l).L(k,j))*obj.maxGradient;
                            end
                        end
                    end
                end
            end
            
            % gradient step
            % loop over layers
            for l = 1:obj.nL+1

                % loop over neurons
                for j = 1:helpingLayer(l)
                    obj.w(l).L(:,j) = obj.w(l).L(:,j) - eta*dw(l).L(:,j);
                end
            end
            
        end % loop over batches
        
        % shuffle data
        newOrder = randperm(obj.N);
        for i = 1:obj.nx
            obj.X(i,:) = obj.X(i,newOrder);
        end
        for i = 1:obj.ny
            obj.Y(i,:) = obj.Y(i,newOrder);
        end
        
    end % loop over epochs
    
end

