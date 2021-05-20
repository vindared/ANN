%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Vorlesung: Maschinelles Lernen in der Regelungstechnik    %
% Lehrstuhl für Regelungstechnik                            %
% Tutor: Hartwig Huber, hartwig.huber@fau.de                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;

% Set parameters:
%% Hier anpassen! Aufgabe e)
nx = 2;
ny = 1;
nNeurons = [1];
batchSize = 1;
eta = 1;
epochs = 1;
adjustEta = 0;

% load data
%% Hier erweitern! Aufgabe b)
% X = ...
% Y = ...

% plot data
%% Hier erweitern! Aufgabe b)


neuronalNet = ANN('nx',2,...
                  'ny',1,...
                  'nL',length(nNeurons),...
                  'nNeurons',nNeurons,...
                  'batchSize',batchSize,...
                  'eta',eta,...
                  'epochs',epochs,...
                  'adjustEta',adjustEta);

%% Hier erweitern!  Aufgabe d)
% neuronalNet.initialize();
% neuronalNet.assignData(X,Y);
% neuronalNet.initWeights('xavier');
% neuronalNet.train();


% Testdata
%% Hier erweitern!  Aufgabe d)
% XTest = ...
% YTest = ...


% feed forward:
%% Hier erweitern!  Aufgabe d)


% plot test data
%% Hier erweitern!  Aufgabe d)

