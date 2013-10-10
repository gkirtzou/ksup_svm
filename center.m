% Centering function
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


function [data mu] = center(data, mu)
% Center the data matrix to have zero mean columns.
% Input : 
%   data - an MxN array with the data to be centered 
%   mu - a M vector with the mean values for each column.
% Output : 
%   data - an MxN array with the data columns centered.
%   mu - a M vector with the mean values for each column. 

    if (~exist('mu','var'))
        mu = mean(data);
    end
    data = data - ones(size(data,1),1)*mu;
end

%% end of file
