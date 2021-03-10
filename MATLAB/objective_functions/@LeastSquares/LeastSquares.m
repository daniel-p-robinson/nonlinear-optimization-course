classdef LeastSquares < handle
%=========================================================================
% This class defines the Least Square objective function given by
%       f(x) = 1/(2m) * ||Ax-b||^2
% where A is an m x n matrix and b is a vector of length m.  There are
% two ways to properly create an instancer of this class:
%       LeastSquares(name)
%       LeastSquares(A,b)
% where name is a string that gives the full path to a .mat file from
% which the matrix A and vector b may be loaded.  For example, one can
% create an instance with the following command:
%       LeastSqures('/Users/danielrobinson/lsprobs/myprob.mat')
% provided the file given by the path exists and has A and b saved in it.
% ========================================================================
% Note: This implementation is not meant for efficiency but rather for
%       clarity. To improve efficiency, one can store Ax-b when performing
%       the function evaluation and reuse it in the gradient computation.
% ========================================================================
    properties
        A          % data matrix
        b          % target vector
        m          % number of samples
        n          % number of features
        x          % saved vector x 
        res        % residual Ax-b at x
        f_computed % 0/1 indicates whether f(x) has been computed
        f          % holds f(x) when f_computed is true
        g_computed % 0/1 indicates whether grad f(x) has been computed
        g          % holds grad f(x) when g_computed is true
        h_computed % 0/1 indicates whether hess f(x) has been computed
        h          % holds Hess f(x) when h_computed is true
        name       % name of this objective function
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
            [self.m,self.n] = size(self.A);
            if self.m <= 0 || self.n <= 0
               error('LeastSquares(ERROR): invalid dimension in A.')  
            end
            if length(self.b) ~= self.m
                error('LeastSquares(ERROR): dimension mismatch in (A,b).')
            end
            self.x   = zeros(self.n,1);
            self.res = -self.b;
            self.f_computed = false;   self.f = [];
            self.g_computed = false;   self.g = [];
            self.h_computed = false;   self.h = [];
            self.name = 'linear-least-squares';
        end
        
        % function evaluation
        function f=func(self, x)
            if length(x) ~= self.n
                error('The input has the wrong dimension.');
            end
            if isequal( x, self.x )
                if self.f_computed
                    f = self.f;
                else
                    f = (1/(2*self.m)) * sum((self.res).^2);
                    self.f = f;
                    self.f_computed = true;
                end
            else
                self.x   = x;
                self.res = self.A*self.x - self.b;
                f        = (1/(2*self.m)) * sum((self.res).^2);
                self.f   = f;   
                self.f_computed = true;
                self.g_computed = false;
                self.h_computed = false;
            end
        end
        
        % gradient evaluation
        function g = grad(self, x)
            if length(x) ~= self.n
                error('The input has the wrong dimension.');
            end
            if isequal( x, self.x )
                if self.g_computed
                    g = self.g;
                else
                    g = (self.A' * self.res) / self.m;
                    self.g = g;
                    self.g_computed = true;
                end
            else
                self.x   = x;
                self.res = self.A*self.x - self.b;
                g = (self.A' * self.res) / self.m;
                self.g = g;
                self.g_computed = true;
                self.f_computed = false;
                self.h_computed = false;
            end
        end
        
        % hessian evaluation is independent of x; keep x for consistency
        function h = hess(self, x) %#ok<INUSD>
            if self.h_computed
                h = self.h;
            else
               h = self.A' * self.A / self.m;
               self.h = h;
               self.h_computed = true;
            end
        end
        
        % hessian-vector product is independent of x; keep x for consistency
        function hv = hessvecprod(self, x, v)
            if ( length(x) ~= self.n ) || ( length(v) ~= self.n )
                error('An input has the wrong dimension.');
            end
            hv = self.A' * (self.A * v) / self.m;
        end
        
    end
end



