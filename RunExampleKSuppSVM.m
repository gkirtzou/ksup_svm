% A toy example of the k-support regularized SVM
%
% Copyright (C) 2013  Matthew Blaschko, Katerina Gkirtzou
%
% This file is part of the ksup-SVM package
% 
% ksup-SVM package is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% ksup-SVM package is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with ksup-SVM.  If not, see <http://www.gnu.org/licenses/>.
%
% If you use this software for scientific work please cite the 
% following two papers
%
% 1) Sparse classification with MRI based markers for neuromuscular disease
%    categoriztion.
%    Gkirtzou Katerina, Deux Jean-François, Bassez Guillaume, Sotiras Aristeidis,
%    Rahmouni Alain, Varacca Thibault, Paragios Nikos and Blaschko B. Matthew
%    4th International Workshop on Machine Learning in Medical Imaging (MLMI), 2013.
%
% 2) Sparse Prediction with the k-Support Norm 
%    Andreas Argyriou, Rina Foygel and Nathan Srebro
%    Neural Information Processing Systems (NIPS), pp. 1466-1474, 2012


%% Load data and packages
clear all;
% Ksupport norm package
addpath(genpath('.'));


%% Create data
% data = a MxN array containing M samples with N-dimensional features
% Generate values from a normal distribution with mean 1 and standard
% deviation 2 as positive class
dataPos = 1 + 2.*randn(100,20);
% Generate values from a normal distribution with mean 1 and standard
% deviation 2 as negative class
dataNeg = 10 + 2.*randn(100,20);
data = [dataPos; dataNeg];
% labels = a Mx1 vector containing the variable that we want to predict
labels = [ ones(100, 1); ones(100, 1)*-1];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialization of variables
% Ksupport norm parameters
lambda = 1
k = 10 % k\in\{1, size(data, 2)\}



%% Training the ksuppport regularized SVM 
% Transductive normalization of data 
[data, mu, d] = normalize(data);
disp(['ksupSVM with lambda=', num2str(lambda), ', k=', num2str(k)]);
% Permute data 
ind = randperm(size(data, 1));
labels = labels(ind);
data = data(ind,:);
% Use 80% for training and the remaining 20% for testing 
threshVal = floor(length(labels)*.8);

% Center the training labels
[labelsTrain, muL] = center(labels(1:threshVal));
% Trainning ksupSVM
[w,cost] = ksupSVM(data(1:threshVal, :), labelsTrain, lambda, k);

%% Validation
pred = data(threshVal+1:end, :)*w;

% Center the validation labels
[labelsVal] = center(labels(threshVal+1:end), muL);
acc = sum(sign(pred) == sign(labelsVal))/length(pred);
disp(['ksupport regularized SVM has accuracy ', num2str(acc)]);

%% end of file




