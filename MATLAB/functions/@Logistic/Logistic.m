classdef Logistic
    % this class defines the Logistic Objective
    % f(x) = 1/m * \sum_{i=1}^n \log( 1+ \exp(-b_ia_i^Tx))
    % where b_i \in \{-1, 1\}, a_i\in R{n}
    
    % Note: This implementation is not meant for efficiency but rather for
    % clarity. To improve the efficiency, one can store intermediate values
    % when perform the function/gradient evaluation and use it later.
    
    properties
        A % data matrix
        b % target vector
        m % number of samples
        n % number of features
    end
    
    methods
        % constructor
        function self=Logistic(varargin)
            if nargin == 1
                dataPath = ['../data/Logistic/', varargin{1},'.mat'];
                if ~(exist(dataPath, 'file') == 2)
                    error(['Logistic(ERROR): either dataset is not in ', dataPath, ...
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
            expterm = exp(-1 * (self.b).*(self.A * x));
            f=1/self.m * sum(log(1 + expterm));
        end
        
        % gradient evaluation
        function g = grad(self, x)
            if size(x,1) ~= size(self.A,2)
                error([inputname(2), ' is of wrong dimension.']);
            end
            expterm = exp(-1 * (self.b).*(self.A * x));
            sigmoid= expterm./(1+expterm);
            g = -1/self.m * ((sigmoid.*self.b)'*self.A)';
        end
        
        % hessian evaluation
        % independent of x, but keep x here for consistency
        function h = hess(self, x)
            if size(x,1) ~= size(self.A,2)
                error([inputname(2), ' is of wrong dimension.']);
            end
            expterm = exp(-1 * (self.b).*(self.A * x));
            sigmoid= expterm./(1+expterm);
            D = diag((1-sigmoid) .* sigmoid);
            h = 1/self.m * self.A' * D * self.A;
        end
        
        % hessian-vector product
        function hv = hessvecprod(self, x, v)
            if size(x,1) ~= size(self.A,2)
                error([inputname(2), ' is of wrong dimension.']);
            end
            if size(v,1) ~= size(self.A,2)
                error([inputname(3), ' is of wrong dimension.']);
            end
            
            expterm = exp(-1 * (self.b).*(self.A * x));
            sigmoid= expterm./(1+expterm);
            temp = self.A * v;
            temp = temp .* (1-sigmoid) .* sigmoid;
            hv = (temp' * self.A)' / self.m;
        end
    end
end



