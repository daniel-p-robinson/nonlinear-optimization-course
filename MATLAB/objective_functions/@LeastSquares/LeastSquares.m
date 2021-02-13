classdef LeastSquares
%=========================================================================
% This class defines the Least Square objective function given by
%       f(x) = 1/(2m) * ||Ax-b||^2
% where A is an m x n matrix and b is a vector of length m.
%
% Note: This implementation is not meant for efficiency but rather for
%       clarity. To improve efficiency, one can store Ax-b when performing
%       the function evaluation and reuse it in the gradient computation.
% ========================================================================
    properties
        A % data matrix
        b % target vector
        m % number of samples
        n % number of features
    end
    
    methods
        % constructor
        function self=LeastSquares(varargin)
            if nargin < 1
               error('LeastSquares(ERROR): no dataset name supplied.') 
            elseif nargin == 1
                if isfile([varargin{1},'.mat'])
                    load([varargin{1},'.mat'],'A','b');
                    self.A = A;
                    self.b = b;
                elseif isfile(varargin{1})
                    load(varargin{1},'A','b');
                    self.A = A;
                    self.b = b;
                else
                    error('LeastSquares(ERROR): dataset cannot be found.')
                end
            else
                self.A = varargin{1};
                self.b = varargin{2};
            end
            self.m = size(self.A,1);
            self.n = size(self.A,2);
        end
        
        % function evaluation
        function f=func(self, x)
            if size(x,1) ~= size(self.A,2)
                error([inputname(2), ' is of wrong dimension.']);
            end
            f= 0.5/self.m * sum((self.A*x - self.b).^2);
        end
        
        % gradient evaluation
        function g = grad(self, x)
            if size(x,1) ~= size(self.A,2)
                error([inputname(2), ' is of wrong dimension.']);
            end
            g = self.A' * (self.A * x - self.b) / self.m;
        end
        
        % hessian evaluation is independent of x; keep x for consistency
        function h = hess(self, x)
            h = self.A' * self.A / self.m;
        end
        
        % hessian-vector product is independent of x; keep x for consistency
        function hv = hessvecprod(self, x, v)
            if size(v,1) ~= size(self.A,2)
                error([inputname(3), ' is of wrong dimension.']);
            end
            hv = self.A' * (self.A * v) / self.m;
        end
        
    end
end



