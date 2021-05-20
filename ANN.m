%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Vorlesung: Maschinelles Lernen in der Regelungstechnik    %
% Lehrstuhl für Regelungstechnik                            %
% Tutor: Hartwig Huber, hartwig.huber@fau.de                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef ANN < handle
    % ANN: artificial neuronal network class (fully connected)
    
    properties
        % net topology
        nx          % (scalar) number of inputs
        ny          % (scalar) number of outputs
        nL          % (scalar) number of hidden layer
        nNeurons    % (vector) number of neurons in hidden layers, 
                    % dimension: (1,nL)
        % net properties (of current data point)
        w           % (vectorial struct) weights in struct representation. Struct 
                    % contains a field (matrix) for every hidden layer as 
                    % well as for the output layer with dimensions:
                    % (nx + 1)
                    % Field names: w(i).L with i = 1 ... nL + 1
                    % size(w(i).L) = [number of inputs + 1, number of neurons]
                    % Bias is the first element in every subvector
        a           % (vectorial struct) activations of last feedforward. Contains 
                    % fields (vector) for every hidden layer and for output 
                    % layer. Vector length is equal to number of neurons in
                    % the respective layer
                    % Field names: w(i).L with i = 1 ... nL + 1
        z           % (vectorial struct) of outputs. Same structure as activations
                    % Field names: z(i).L with i = 1 ... nL + 1
        delta       % (vectorial struct) of errors.  Same structure as activations
                    % Field names: delta(i).L with i = 1 ... nL + 1
        % information
        J           % (vector) progression of cost averaged over batches
        % data
        N           % (scalar) number of data points (training data)
        X           % (matrix) input data points, dimension: (nx,N)
        Y           % (matrix) output data points, dimension: (ny,N)
        % parameters
        batchSize   % (scalar) number of samples to be used in one batch
        eta         % (scalar) learning rate or initial learning rate
        epochs      % (scalar) number of epochs
        phiType     % type of neuron-activation function
        hType       % type of output-activation function
        adjustEta   % (scalar) if 0, learning rate is constant
        clipLoss    % (scalar) if 0, cost function is not clipped (logloss)
        maxGradient % maximum value of single partial derivative
    end
    
    methods
        %% Constructor
        function obj = ANN(varargin)
            % Constuructor for ANN
            
            arglist = properties(obj);
            storage=struct;
            written={};
            found = 0;
            
            % Input parser
            % Looping through the 'varargin' list. Prints errors is case of
            % wrong format of arguments.
            if ~isempty(varargin)
				if mod(length(varargin),2) % uneven number
					error('When passing additional arguments, you must pass them pairwise!')
				end
				for index = 1:2:length(varargin) % loop through all passed arguments:
					for arg = 1:length(arglist)
						if strcmp(varargin{index},arglist{arg})
							storage.(arglist{arg}) = varargin{index+1};
							written=[written arglist{arg}]; %#ok<AGROW>
							found=1;
							break
						end 
					end % for arg
					% argument wasnt found in for-loop
					if ~found
						error([varargin{index} ' is not a valid property of class opdeo_sys!']);
					end
					found=0; % reset found
				end % for index
            end % if isempty varargin
            
            % Check inputs (roughly)
            
            discList = {'nx','ny','nNeurons','nL'}; % necessary fields
            for parname = discList
                if ~isfield(storage,parname{1})
                    error(['The field "' parname{1} '" must be provided!']);
                end
            end
            
            % check input number
            if (storage.nx < 1)
                error('Net must have at least one input!');
            end
            % check output number
            if (storage.ny < 1)
                error('Net must have at least one output!');
            end
            % check hidden layer number
            if (storage.nL < 0)
                error('Number of hidden layers cant be negative!');
            end
            % check neuron number
            if (size(storage.nNeurons,1) ~= 1)
                % try transpose
                if  (size(storage.nNeurons,2) ~= 1)
                    error('"nNeurons" must be of appropriate dimesnion!');
                else
                    storage.nL = storage.nL.';
                end
            end
            if(size(storage.nNeurons,2) ~= storage.nL)
                error('"nL" and "nNeurons" must be consistent!');
            end
            
            % if data is provided: check it
            discList = {'X','Y'}; % data fields
            status = 0;
            for parname = discList
                if isfield(storage,parname{1})
                    status = status + 1;
                end
            end
            
            if (isfield(storage,'X') && isfield(storage,'Y'))
                if(size(storage.X,1) ~= nx)
                    error('X must be consistent to nx!');
                end
                if(size(storage.Y,1) ~= ny)
                    error('Y must be consistent to ny!');
                end
                if(size(storage.Y,2) ~= size(storage.X,2))
                    error('Y and X must be consistent!');
                end
                if isfield(storage,'N')
                    if(size(storage.Y,2) ~= storage.N)
                        error('Y and X must be consistent to N!');
                    end
                else
                    storage.N = size(storage.Y,2);
                end
            elseif (isfield(storage,'X') || isfield(storage,'Y'))
                error('You must X and Y when directly assigning data!');
            end
            
            % check params
            if (isfield(storage,'batchSize'))
                if (storage.batchSize < 1)
                    error('batchSize must be at least 1!');
                end
            else
                storage.batchSize = 1;
            end
            if (isfield(storage,'eta'))
                if (storage.eta < 0)
                    error('eta must be greater than 0!');
                end
            else
                storage.eta = 1;
            end
            if (isfield(storage,'epochs'))
                if (storage.epochs < 1)
                    error('epochs must be at least 1!');
                end
            else
                storage.epochs = 1;
            end
            if ~isfield(storage,'phiType')
                storage.phiType = 'sigmoid';
            end
            if ~isfield(storage,'hType')
                storage.hType = 'sigmoid';
            end
            if ~isfield(storage,'clipLoss')
                storage.clipLoss = 0;
            end
            if ~ isfield(storage,'maxGradient')
                storage.maxGradient = 50;
            end
            % Loop through 'storage' and write the values to the object.
            fn = fieldnames(storage);
			for fn_idx = 1:length(fn)
				obj.(fn{fn_idx})=storage.(fn{fn_idx});
			end
            
        end % constructor end
        
    end
    
    
    
    methods
        %% Remaining methods
        initialize(obj); % 'allocate' memory
        assignData(obj,X,Y); % assign training data
        initWeights(obj,type); % initialize weights
        y = feedForward(obj,x); % feed forward pass data
        z = activation(obj,a,output); % activation function
        dz = derivedActivation(obj,a,output); % activation function
        J = costFunction(obj,yReal,yNet); % cost (loss) function used for optimization
        dJ = derivedCostFunction(obj,yReal,yNet); % derivative of cost (loss) function
        backpropagate(obj,yReal); % performs backpropagation for a single input output sample
        train(obj); % trains the ANN using the assigned data
    end
end










