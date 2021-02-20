classdef LeastSquaresTukey
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
        A % data matrix
        b % target vector
        m % number of samples
        n % number of features
        d % constant value
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
            self.m = size(self.A,1);
            self.n = size(self.A,2);
            self.d = 4.685;  % assign value from literature to d
        end
        
        % function evaluation
        function f=func(self, x)
            if size(x,1) ~= size(self.A,2)
                error([inputname(2), ' is of wrong dimension.']);
            end
            u = (self.A*x - self.b); % input values for Tukey bisquare loss function
            f=0; 
            for i = 1:self.m
                if abs(u(i))<=self.d %first possibility of piecewise Tukey bisquare loss function
                    f=f+(((self.d).^2)/6)*(1-(1-(u(i)/self.d)^2)^3);
                else
                    f=f+((self.d)^2)/6; % a constant value
                end    
            end
        end
        
        % gradient evaluation
        function g = grad(self, x)
            if size(x,1) ~= size(self.A,2)
                error([inputname(2), ' is of wrong dimension.']);
            end
            u = (self.A*x - self.b); % input into derivative of Tukey bisquare loss function
            s=zeros([self.m 1]);     % initializes array to hold differentiation information
            for i = 1:self.m 
                if abs(u(i))<=self.d
                    s(i)=u(i)*((1-(u(i)/self.d)^2)^2); % first possibility of piecewise Tukey bisquare loss function
                else
                    s(i)=0; % zero from differentiating a constant
                end    
            end
            g= self.A' * s;  % chain rule gives answer
        end
        
        % hessian evaluation is independent of x; keep x for consistency
        function h = hess(self, x)
            u  = (self.A*x - self.b); % input into second derivative of Tukey bisquare loss function
            sd = zeros(self.m,1);     % hold initializes array to hold differentiation information
            for i = 1:self.m
                if abs(u(i))<=self.d 
                    f = ( u(i)/self.d )^2;
                    sd(i)=(1-f)*(1-5*f);
                else
                    sd(i)=0; 
                end    
            end
            h = (self.A)'*diag(sd)*(self.A);
        end
        
        % hessian-vector product is independent of x; keep x for consistency
        function hv = hessvecprod(self, x, v)
            if size(x,1) ~= size(self.A,2)
                error([inputname(2), ' is of wrong dimension.']);
            end
            if size(v,1) ~= size(self.A,2)
                error([inputname(3), ' is of wrong dimension.']);
            end
            
            %this part recalculates the hessian matrix determined above so
            %that multiplication may be done below
            h=zeros(self.n);
            AT=self.A';
            
            u = (self.A*x - self.b);
            sd=zeros([self.m 1]);
            for i = 1:self.m
                if abs(u(i))<=self.d
                    sd(i)=(1/((self.d).^4))*((u(i)).^2-(self.d).^2)*(5*(u(i)).^2-(self.d).^2);
                else
                    sd(i)=0;
                end    
            end
            
            for k=1:self.n
                for j=1:self.n
                    for i=1:self.m
                        h(k,j)=h(k,j)+AT(k,i)*AT(j,i)*sd(i);
                    end
                end
            end
            
            %multiplies the recalculated hessian matrix by the vector
            hv = h * v;
        end
        
    end
end