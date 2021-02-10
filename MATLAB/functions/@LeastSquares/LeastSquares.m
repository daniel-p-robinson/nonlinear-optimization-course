classdef LeastSquares
    % this class defines the Least Square Objective
    % f(x) = 1/(2m) * ||Ax-b||^2 where A\in R^{m\times n} and b\in R^{m}
    
    % Note: This implementation is not meant for efficiency but rather for
    % clarity. To improve the efficiency, one can store Ax-b when perform
    % the function evaluation and use it later in the gradeint computation
    properties
        A % data matrix
        b % target vector
        m % number of samples
        n % number of features
    end
    
    methods
        % constructor
        function self=LeastSquares(varargin)
            if nargin == 1
                dataPath = ['../data/LeastSquares/', varargin{1},'.mat'];
                if ~(exist(dataPath, 'file') == 2)
                    error(['LeastSquares(ERROR): either dataset is not in ', dataPath, ...
                        ' or the dataset cannot be used for logistic loss.'])
                end
                load(dataPath, 'A', 'b');
                self.A = A;
                self.b = b;
            end
            if nargin == 2
                if isa(varargin{1}, 'char')
                    dataPath = [varargin{1}, varargin{2},'.mat'];
                    if ~(exist(dataPath, 'file') == 2)
                    error(['Logistic(ERROR): either dataset is not in ', dataPath, ...
                        ' or the dataset cannot be used for logistic loss.'])
                    end
                    load(dataPath, 'A', 'b');
                    self.A = A;
                    self.b = b;
                else
                    self.A = varargin{1};
                    self.b = varargin{2};
                end
                
            end
            self.m = size(self.A, 1);
            self.n = size(self.A, 2);
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
        
        % hessian evaluation
        % independent of x, but keep x here for consistency
        function h = hess(self, x)
            h = self.A' * self.A / self.m;
        end
        
        % hessian-vector product
        % independent of x, but keep x here for consistency
        function hv = hessvecprod(self, x, v)
            if size(v,1) ~= size(self.A,2)
                error([inputname(3), ' is of wrong dimension.']);
            end
            hv = self.A' * (self.A * v) / self.m;
        end
    end
end



