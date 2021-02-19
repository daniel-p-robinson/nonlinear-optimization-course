classdef Logistic
% ========================================================================
% This class defines the logistic objective function given by
%       f(x) = 1/m * \sum_{i=1}^n \log( 1+ \exp(-b_ia_i^Tx))
% where b_i is in {-1,1} and a_i is a vector in R^n.
%    
% Note: This implementation is not meant for efficiency but rather for
%       clarity. To improve efficiency, one can store intermediate values
%       when perform the function/gradient evaluation and reuse it later.
% ========================================================================    
    properties
        A % data matrix
        b % target vector
        m % number of samples
        n % number of features
    end
    
    methods
        % constructor
        function self=Logistic(varargin)
            if nargin < 1
               error('Logistic(ERROR): no dataset name supplied.') 
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
                    error('Logistic(ERROR): dataset cannot be found.')
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
            expterm = exp(-1 * (self.b).*(self.A * x));
            % safeguard
            if sum(isinf(expterm)) ~=0
                warning('Logistic.func: expterm has Inf value')
            end
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



