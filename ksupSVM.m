% The k-support regularized Support Vector Machine
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

function [w,costs] = ksupSVM(X,Y,lambda,k,w0,h, ...
                              iters_acc,eps_acc);
% Compute the k-support reguralized SVM for a set on input samples.
% The first 4 arguments are required!
% Input : X - a MxN matrix of input samples, where M is the number
%             of samples and N is the number of dimension per
%             sample.
%         Y - a Mx1 matrix of the output samples.
%         lambda - the scalar parameter that specifies the weight on
%             the k-support penalty on the learned coefficients.
%         k - the scalar parameter of k support penalty that
%             negative correlates with the degree of sparsity of the
%             learned coefficients, where k \in \{1, N\}.
%         w0 - the initial values of the learned coefficients. The
%             default value is zero.
%         h - the Huber parameter. The default value is 0.1 
%             For more details see e.g. Olivier Chapelle. 
%             Training a Support Vector Machine in the Primal,
%             Neural Computation, 2007. Eq. (18).
%         iters_acc - the maximum # of iterations as a termination
%             criterion for the ksupport regularization if the
%             tolerance criterion isn't met. The default
%             value is 2000.
%         eps_acc - the tolerance used as termination criterion for
%             the minumization of the k-support regularization. The
%             default value is 1e-4.
% Output : w - the learned coefficents calculated by the ksupSVM.
%          costs - the final cost.
%
    if(nargin<8)
        eps_acc = 1e-4;
    end

    if(nargin<7)
        iters_acc = 2000; 
    end
    
    if(nargin<6) 
        h = 0.1;
    end
    
    if(nargin<5)
        w0 = zeros(size(X,2),1);
    end
           
    % the lipschitz constant for gradient of  hinge loss
    if(size(X,1)>size(X,2)) 
        L = eigs(X'*X,1)/(2*h);
    else
        L = eigs(X*X',1)/(2*h);
    end
    [w,costs] = overlap_nest(@(w)(huberLoss(w,X,Y,h)),...
                             @(w)(gradHuberLoss(w,X,Y,h)), lambda, ...
                             L, w0, k, iters_acc,eps_acc);
end

function [ind1,ind2] = huberInd(w,X,Y,h);
% Indices for the Huber approximation of the Hinge loss 
% Input : w - the learned coefficients.
%         X - a MxN matrix of input samples, where M is the number
%             of samples and N is the number of dimension per sample.
%         Y - a Mx1 matrix of the output samples.
%         h - the Huber parameter. For more details see e.g. Olivier Chapelle. 
%             Training a Support Vector Machine in the Primal,
%             Neural Computation, 2007. Eq. (18).
% Output : ind1 - the indices for the Huber approximation of the
%                  Hinge loss where the margin is smaller that 1-h.
%          ind2 - the indices for the Huber approximation of the
%                 Hinge loss where the absolute 1-margin is smaller
%                 or equal to h.

  margin = Y.*(X*w);
  ind1 = find(margin<1-h);
  ind2 = find(abs(1-margin)<=h);
end


function l = huberLoss(w,X,Y,h)
% The Huber approximation of the Hinge loss 
% Input : w - the learned coefficients.
%         X - a MxN matrix of input samples, where M is the number
%             of samples and N is the number of dimension per sample.
%         Y - a Mx1 matrix of the output samples.
%         h - the Huber parameter. For more details see e.g. Olivier Chapelle. 
%             Training a Support Vector Machine in the Primal,
%             Neural Computation, 2007. Eq. (18).
% Output l - the Hinge loss using the Huber approximation.
  
    [ind1,ind2] = huberInd(w,X,Y,h);
    l = 0;
    if(length(ind1)>0)
        l = sum(1-Y(ind1).*(X(ind1,:)*w));
    end
    l2 = 0;
    if(length(ind2)>0)
        l2 = sum((1+h-Y(ind2).*(X(ind2,:)*w)).^2)./(4*h);
    end
    l = l+l2;
end

function g = gradHuberLoss(w,X,Y,h)
% The gradient of the Huber approximation of the hinge loss 
% Input : w - the learned coefficients.
%         X - a MxN matrix of input samples, where M is the number
%             of samples and N is the number of dimension per sample,
%         Y - a Mx1 matrix of the output samples,
%         h - the Huber parameter. For more details see e.g. Olivier Chapelle. 
%             Training a Support Vector Machine in the Primal,
%             Neural Computation, 2007. Eq. (18)
% Output : g - the gradient of Hinge loss usign the Huber approximation
  
    [ind1,ind2] = huberInd(w,X,Y,h);
    g = zeros(size(w));
    if(length(ind1)>0)
        g = g - X(ind1,:)'*Y(ind1);
    end
    if(length(ind2)>0)
        g = g + (X(ind2,:)'*(X(ind2,:)*w) - (1+h)*X(ind2,:)'*Y(ind2))./(2*h);
    end
end

% end of file

