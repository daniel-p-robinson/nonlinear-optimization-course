classdef LinearSphere < handle
% =========================================================================
% Computes the objective function, the objective function gradient,
% the constraint, the constraint Jacobian, and the Hessian of the
% Lagrangian function for the optimization problem
%
%          (NEP)  minimize   f(x) = x1 + x2
%                subject to  c(x) = x1^2 + x2^2 - 2 = 0
%
% where x = (x1,x2).
%=========================================================================
    % List of properties needed for the class.
    properties 
         n          % number of optimization variables
         m          % number of constraints in the optimization problem
         x          % saved primal vector x 
         y          % saved dual vector y
         f_computed % 0/1 indicates whether f(x) has been computed
         f          % holds f(x) when f_computed is true
         g_computed % 0/1 indicates whether grad f(x) has been computed
         g          % holds grad f(x) when g_computed is true
         c_computed % 0/1 indicates whether c(x) has been computed         
         c          % holds c(x) when c_computed is true
         j_computed % 0/1 indicates whether Jac c(x) has been computed
         j          % holds the Jacobian of c(x) when J_computed is true
         h_computed % 0/1 indicates whether Hessian of Lagrangian has been computed
         h          % holds Hess of Lagrangian when h_computed is true
%         mE        % number of equality constraints: c_E(x) = 0
%         mI        % number of inequality constraints: cl <= c_I(x) <= cu
%         cL        % lower bound vector for inequality constraints
%         cU        % upper bound vector for inequality constraints
%         xL        % lower bound vector for optimization variables: xL <= x
%         xU        % upper bound vector for optimization variables: x <= xU
%         x0        % initial solution estimate
%         yE0       % initial Lagrange multiplier for cE(x) = 0.
%         yI0       % initial Lagrange multiplier for cL <=  cI(x) <= cU.
%         obj_type  % type of objective function:     linear, quadratic, linear-least-squares, nonlinear-least-squares, general
%         conE_type % type of equality constraints:   linear, quadratic, general
%         conI_type % type of inequality constraints: linear, quadratic, general
         name      % name associated to the optimization problem               
    end
    
    methods
        % constructor
        function self=LinearSphere(varargin)
            self.n          = 2;
            self.m          = 1;
            self.x          = zeros(self.n,1);
            self.y          = zeros(self.m,1);
            self.f_computed = false;  self.f = [];
            self.g_computed = false;  self.g = [];
            self.c_computed = false;  self.c = [];
            self.j_computed = false;  self.j = [];
            self.h_computed = false;  self.h = [];
            self.name       = 'LinearSphere';
%             self.mE  = 0;
%             self.mI  = 0;
%             self.cL  = [];
%             self.cU  = [];
%             self.xL  = [-inf;-inf];
%             self.xU  = [inf;inf];
%             self.x0  = [0;0];
%             self.yE0 = [];
%             self.yI0 = [];
%             self.obj_type  = 'general';
%             self.conE_type = [];
%             self.conI_type = [];
        end
        
        % Evaluate objective function f at x.
        function f = func(self, x)
            if length(x) ~= self.n
                error('Input must be a %g dimensional vector!',self.n);
            end
            if isequal( x, self.x )
                if self.f_computed
                    f = self.f;
                else
                    f = x(1)+x(2);
                    self.f = f;
                    self.f_computed = true;
                end
            else
                self.x = x;
                f      = x(1)+x(2);
                self.f = f;   
                self.f_computed = true;
                self.g_computed = false;
                self.c_computed = false;
                self.j_computed = false;
                self.h_computed = false;
            end
        end
        
        % Evaluate the gradient of f at x.            
        function g = grad(self, x)
            if length(x) ~= self.n
                error('Input must be a %g dimensional vector!',self.n);
            end
            if isequal( x, self.x )
                if self.g_computed
                    g = self.g;
                else
                    g = [1;1];
                    self.g = g;
                    self.g_computed = true;
                end
            else
                self.x = x;
                g      = [1;1];
                self.g = g;
                self.f_computed = false;               
                self.g_computed = true;
                self.c_computed = false;
                self.j_computed = false;
                self.h_computed = false;
            end
        end
           
        % Evaluate constraint function c at x.
        function c = constraint(self, x)
            if length(x) ~= self.n
                error('Input must be a %g dimensional vector!',self.n);
            end
            if isequal( x, self.x )
                if self.c_computed
                    c = self.c;
                else
                    c = x(1)^2 + x(2)^2 - 2;
                    self.c = c;
                    self.c_computed = true;
                end
            else
                self.x = x;
                c      = x(1)^2 + x(2)^2 - 2;
                self.c = c;   
                self.f_computed = false;
                self.g_computed = false;
                self.c_computed = true;
                self.j_computed = false;
                self.h_computed = false;
            end
        end
        
        % Evaluate the Jadobian of the constraint at x.            
        function j = jacobian(self, x)
            if length(x) ~= self.n
                error('Input must be a %g dimensional vector!',self.n);
            end
            if isequal( x, self.x )
                if self.j_computed
                    j = self.j;
                else
                    j = [ 2*x(1) 2*x(2) ] ;
                    self.j = j;
                    self.j_computed = true;
                end
            else
                self.x = x;
                j      = [ 2*x(1) 2*x(2) ];
                self.j = j;
                self.f_computed = false;
                self.g_computed = false;
                self.c_computed = false;
                self.j_computed = true;
                self.h_computed = false;
            end
        end
        
        % Evaluate the Hessian of f at x.
        function h = hess(self,x,y)
            if length(x) ~= self.n
                error('First input must be a %g dimensional vector!',self.n);
            end
            if length(y) ~= self.m
                error('Second input must be a %g dimensional vector!',self.m);
            end 
            if isequal(x,self.x) && isequal(y,self.y)
                if self.h_computed
                    h = self.h;
                else
                    h = -2*y*eye(2);
                    self.h = h;
                    self.h_computed = true;
                end
            else
                self.x = x;
                self.y = y;
                h      = -2*y*eye(2);
                self.h = h;
                self.f_computed = false;
                self.g_computed = false;
                self.c_computed = false;
                self.j_computed = false;
                self.h_computed = true;
            end
        end
        
        % Compute product of Hessian of Lagrangian at (x,y) with vector x.
        function hv = hessvecprod(self, x, y, v)
            if length(x) ~= self.n
                error('First input must be a %g dimensional vector!',self.n);
            end
            if length(y) ~= self.m
                error('Second input must be a %g dimensional vector!',self.m);
            end
            if length(v) ~= self.n
                error('Third input must be a %g dimensional vector!',self.n);
            end
            if isequal(x,self.x) && isequal(y,self.y)
                if self.h_computed
                   hv = self.h*v;
                else
                    self.h = -2*y*eye(2,2);
                    self.h_computed = true;
                    hv = self.h*v;
                end
            else
                self.x = x;
                self.y = y;
                self.h = -2*y*eye(2,2); 
                self.f_computed = false;
                self.g_computed = false;
                self.c_computed = false;
                self.j_computed = false;
                self.h_computed = true;
                hv = self.h*v;
            end
        end
    end
end