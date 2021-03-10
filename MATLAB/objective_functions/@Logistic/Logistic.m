classdef Logistic < handle
% ========================================================================
% This class defines the logistic objective function given by
%       f(x) = (1/m)*sum_{i=1}^m log( 1+ exp(-b_i a_i^Tx) )
% where b_i is in {-1,1} and a_i is a vector in R^n such that the ith
% column of the data matrix A is a_i^T. 
% ========================================================================   
% Note: This implementation is not meant for efficiency but rather for
%       clarity. To improve efficiency, one can store intermediate values
%       when perform the function/gradient evaluation and reuse it later.
% ========================================================================    
    properties
        A          % data matrix
        b          % target vector
        m          % number of samples
        n          % number of features
        x          % saved x
        expterm    % 
        sigmoid    %
        f_computed % 0/1 indicates whether f(x) has been computed
        f          % holds f(x) when f_computed is true
        g_computed % 0/1 indicates whether grad f(x) has been computed
        g          % holds grad f(x) when g_computed is true
        h_computed % 0/1 indicates whether hess f(x) has been computed
        h          % holds hess f(x) when h_computed is true
        name       % name of the objective function
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
            [self.m,self.n] = size(self.A);
            if self.m <= 0 || self.n <= 0
                error('Logistic(ERROR): invalid dimensions in A.')
            end
            if length(self.b) ~= self.m
                error('Logistic(ERROR): invalid dimensions in A.')
            end
            self.x          = zeros(self.n,1);
            self.expterm    = ones(self.m,1);
            self.sigmoid    = 0.5*ones(self.m,1);
            self.f_computed = false;   self.f = [];
            self.g_computed = false;   self.g = [];
            self.h_computed = false;   self.h = [];
            self.name       = 'Logistic';
        end
        
        % function evaluation
        function f=func(self, x)
            if length(x) ~= self.n
                error('Input must be a %g dimensional vector!',self.n);
            end
            if isequal( x, self.x )
                if self.f_computed
                    f = self.f;
                else
                    f = 1/self.m * sum(log(1 + self.expterm));
                    self.f = f;
                    self.f_computed = true;
                end
            else
                self.x = x;
                self.expterm = exp(-1*(self.b).*(self.A * self.x));
                self.sigmoid = self.expterm./(1+self.expterm);
                f = 1/self.m * sum(log(1 + self.expterm));
                self.f = f;
                self.f_computed = true;
                self.g_computed = false;
                self.h_computed = false;
            end
        end
        
        % gradient evaluation
        function g = grad(self, x)
            if length(x) ~= self.n
                error('Input must be a %g dimensional vector!',self.n);
            end
            if isequal( x, self.x )
                if self.g_computed
                    g = self.g;
                else
                    g = (-1/self.m) * ((self.sigmoid.*self.b)'*self.A)';
                    self.g = g;
                    self.g_computed = true;
                end
            else
                self.x = x;
                self.expterm = exp(-1*(self.b).*(self.A * self.x));
                self.sigmoid = self.expterm./(1+self.expterm);
                g = (-1/self.m) * ((self.sigmoid.*self.b)'*self.A)';
                self.g = g;
                self.g_computed = true;
                self.f_computed = false;
                self.h_computed = false;
            end
        end
        
        % hessian evaluation
        function h = hess(self, x)
            if length(x) ~= self.n
                error('Input must be a %g dimensional vector!',self.n);
            end
            if isequal( x, self.x )
                if self.h_computed
                    h = self.h;
                else
                    D = diag((1-self.sigmoid) .* self.sigmoid);
                    h = (1/self.m) * self.A' * D * self.A;
                    self.h = h;
                    self.h_computed = true;
                end
            else
                self.x = x;
                self.expterm = exp(-1*(self.b).*(self.A * self.x));
                self.sigmoid = self.expterm./(1+self.expterm);
                D = diag((1-self.sigmoid) .* self.sigmoid);
                h = (1/self.m) * self.A' * D * self.A;
                self.h = h;
                self.h_computed = true;
                self.f_computed = false;
                self.g_computed = false;
            end
        end
        
        % hessian-vector product
        function hv = hessvecprod(self, x, v)
            if ( length(x) ~= self.n ) || ( length(v)~= self.n )
                error('An input has the wrong dimension.');
            end
            if isequal( x, self.x )
                temp = self.A * v;
                temp = temp .* (1-self.sigmoid) .* self.sigmoid;
                hv   = (temp' * self.A)' / self.m;
            else
               self.x = x;
               self.expterm = exp(-1 * (self.b).*(self.A * self.x));
               self.sigmoid= self.expterm./(1+self.expterm);
               temp = self.A * v;
               temp = temp .* (1-self.sigmoid) .* self.sigmoid;
               hv   = (temp' * self.A)' / self.m;
               self.h_computed = false;
               self.f_computed = false;
               self.g_computed = false;
            end
        end
        
    end
end



