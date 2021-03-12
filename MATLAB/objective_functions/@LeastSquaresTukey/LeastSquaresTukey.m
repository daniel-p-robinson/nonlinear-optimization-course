classdef LeastSquaresTukey < handle
%=========================================================================
% This class defines the robust linear regression objective function using 
% Tukey's bisquare loss function. The objective function is given by
%
%    f(x) = sum_{i=1}^m rho(a_i^Tx - b_i) 
%
% where A is an m x n matrix, b is a vector of length m, and rho is
%
%    rho(u) = d^2/6 (1 - (1-(u/d)^2)^3)   if  |u| <= d 
%             d^2/6                       if  |u| >  d
%
% for some chosen parameter d (default value is 4.685). The default value
% for d is based on the recommendation found in the following:
%
%    L. Chang, S. Roberts, and A. Wlsh (2018), "Robust Lasso Regression
%    Using Tukey's Biweight Criterion,", Technometrics, 60, 36-47.
% ========================================================================
    properties
        A          % data matrix
        b          % target vector
        m          % number of samples
        n          % number of features
        d          % constant value
        x          % saved vector x
        res        % residual at x: res = Ax-b
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
        function self=LeastSquaresTukey(varargin)
            if nargin < 1
               error('LeastSquaresTukey(ERROR): no dataset name supplied.') 
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
                    error('LeastSquaresTukey(ERROR): dataset cannot be found.')
                end
            else
                self.A = varargin{1};
                self.b = varargin{2};
            end
            [self.m,self.n] = size(self.A);
            if self.m <= 0 || self.n <= 0
               error('LeastSquaresTukey(ERROR): invalid dimensions for A.')
            end
            if length(b) ~= self.m
               error('LeastSquaresTukey(ERROR): invalid dimensions in (A,b).')
            end
            self.d   = 4.685; % value from the literature
            self.x   = zeros(self.n,1);
            self.res = -self.b;
            self.f_computed = false;   self.f = [];
            self.g_computed = false;   self.g = [];
            self.h_computed = false;   self.h = [];
            self.name       = 'nonlinear-least-squares-tukey';
        end
        
        % function evaluation
        function f=func(self, x)
            if length(x) ~= self.n
                error('Input has the wrong dimension.');
            end
            if isequal( x, self.x )
                if self.f_computed
                    f = self.f;
                else
                    f = sum((self.d^2/6)*min(1-(1-(self.res/self.d).^2).^3, 1));
                    self.f = f;
                    self.f_computed = true;
                end
            else
                self.x = x;
                self.res = (self.A*self.x - self.b); 
                f = sum((self.d^2/6)*min(1-(1-(self.res/self.d).^2).^3, 1));
                self.f = f;
                self.f_computed = true;
                self.g_computed = false;
                self.h_computed = false;
            end
        end
        
        % gradient evaluation
        function g = grad(self, x)
            if length(x) ~= self.n
                error('Input has the wrong dimension.');
            end
            if isequal( x, self.x )
                if self.g_computed
                    g = self.g;
                else
                    g = self.A' * (self.res.*(1-(self.res/self.d).^2).^2.*(abs(self.res) <= self.d));  % chain rule gives answer
                    self.g = g;
                    self.g_computed = true;
                end
            else
                self.x   = x;
                self.res = (self.A*self.x - self.b);
                g = self.A' * (self.res.*(1-(self.res/self.d).^2).^2.*(abs(self.res) <= self.d));  % chain rule gives answer
                self.g = g;
                self.g_computed = true;
                self.f_computed = false;
                self.h_computed = false;
            end
        end
        
        % hessian evaluation is independent of x; keep x for consistency
        function h = hess(self, x)
            if length(x) ~= self.n
                error('Input must be a %g dimensional vector!',self.n);
            end
            if isequal( x, self.x )
                if self.h_computed
                    h = self.h;
                else
                    sd = (1-(self.res/self.d).^2).*(1-5*(self.res/self.d).^2).*(abs(self.res) <= self.d);
                    h  = (self.A)'*spdiags(sd,0,self.m,self.m)*(self.A);
                    self.h = h;
                    self.h_computed = true;
                end
            else
                self.x = x;
                self.res = (self.A*self.xx - self.b); 
                sd = (1-(self.res/self.d).^2).*(1-5*(self.res/self.d).^2).*(abs(self.res) <= self.d);
                h = (self.A)'*spdiags(sd,0,self.m,self.m)*(self.A);
                self.h = h;
                self.h_computed = true;
                self.f_computed = false;
                self.g_computed = false;
            end
        end
        
        % hessian-vector product is independent of x; keep x for consistency
        function hv = hessvecprod(self, x, v)
            if length(x) ~= self.n
                error('First input has the wrong dimension.');
            end
            if length(v) ~= self.n
                error('Second input has the wrong dimension.');
            end
            if isequal( x, self.x )
                sd = (1-(self.res/self.d).^2).*(1-5*(self.res/self.d).^2).*(abs(self.res) <= self.d);
                hv = (self.A)'*spdiags(sd,0,self.m,self.m)*(self.A)*v;
            else
               self.x   = x;
               self.res = (self.A*self.x - self.b);
               sd = (1-(self.res/self.d).^2).*(1-5*(self.res/self.d).^2).*(abs(self.res) <= self.d);
               hv = (self.A)'*spdiags(sd,0,self.m,self.m)*(self.A)*v;
               self.h_computed = false;
               self.f_computed = false;
               self.g_computed = false;
            end
        end
    end
end