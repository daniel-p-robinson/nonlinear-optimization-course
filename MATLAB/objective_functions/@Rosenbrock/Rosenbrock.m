classdef Rosenbrock < handle
    % This class defines the following Rosenbrock function:
    % f(x1,x2) = 100(x2 - x1^2)^2 + (1-x1)^2
    
    % List of properties needed for the class.
    properties 
        n          % number of optimization variables
        mE         % number of equality constraints: c_E(x) = 0
        mI         % number of inequality constraints: cl <= c_I(x) <= cu
        cL         % lower bound vector for inequality constraints
        cU         % upper bound vector for inequality constraints
        xL         % lower bound vector for optimization variables: xL <= x
        xU         % upper bound vector for optimization variables: x <= xU
        x0         % initial solution estimate
        yE0        % initial Lagrange multiplier for cE(x) = 0.
        yI0        % initial Lagrange multiplier for cL <=  cI(x) <= cU.
        obj_type   % type of objective function:     linear, quadratic, linear-least-squares, nonlinear-least-squares, general
        conE_type  % type of equality constraints:   linear, quadratic, general
        conI_type  % type of inequality constraints: linear, quadratic, general
        name       % name associated to the optimization problem  
        x          % saved vector x 
        f_computed % 0/1 indicates whether f(x) has been computed
        f          % holds f(x) when f_computed is true
        g_computed % 0/1 indicates whether grad f(x) has been computed
        g          % holds grad f(x) when g_computed is true
        h_computed % 0/1 indicates whether hess f(x) has been computed
        h          % holds Hess f(x) when h_computed is true
    end
    
    methods
        % constructor
        function self=Rosenbrock(varargin)
            self.n   = 2;
            self.mE  = 0;
            self.mI  = 0;
            self.cL  = [];
            self.cU  = [];
            self.xL  = [-inf;-inf];
            self.xU  = [inf;inf];
            self.x0  = [0;0];
            self.yE0 = [];
            self.yI0 = [];
            self.obj_type  = 'general';
            self.conE_type = [];
            self.conI_type = [];
            self.name      = 'Rosenbrock';
            self.x          = zeros(self.n,1);
            self.f_computed = false;  self.f = [];
            self.g_computed = false;  self.g = [];
            self.h_computed = false;  self.h = [];
        end
        
        % Evaluate objective function f at x.
        function f = func(self, x)
            if length(x) ~= self.n
                error('Input must be a %g dimensional vector!',self.n);
            end
            x1 = x(1);
            x2 = x(2);
            if isequal( x, self.x )
                if self.f_computed
                    f = self.f;
                else
                    f = 100*(x2-x1^2)^2 + (1-x1)^2;
                    self.f = f;
                    self.f_computed = true;
                end
            else
                self.x = x;
                 f = 100*(x2-x1^2)^2 + (1-x1)^2;
                self.f = f;   
                self.f_computed = true;
                self.g_computed = false;
                self.h_computed = false;
            end
        end
        
        % Evaluate the gradient of f at x.            
        function g = grad(self, x)
            if size(x,1) ~= 2
                error('Input must be a 2 dimensional vector!');
            end
            x1 = x(1);
            x2 = x(2);
            if isequal( x, self.x )
                if self.g_computed
                    g = self.g;
                else
                    g = [-400*x1*(x2-x1^2) - 2*(1-x1) ;
                          200*(x2-x1^2)              ];
                    self.g = g;
                    self.g_computed = true;
                end
            else
                self.x = x;
                g = [-400*x1*(x2-x1^2) - 2*(1-x1) ;
                      200*(x2-x1^2)              ];
                self.g = g;
                self.g_computed = true;
                self.f_computed = false;
                self.h_computed = false;
            end
        end
        
        % Evaluate the Hessian of f at x.
        function h = hess(self, x)
            if length(x) ~= 2
                error('Input must be a 2 dimensional vector!');
            end
            x1 = x(1);
            x2 = x(2);
            if isequal( x, self.x )
                if self.h_computed
                    h = self.h;
                else
                    h = [-400*(x2-3*x1^2)+2 , -400*x1 ;
                         -400*x1            ,   200   ];
                    self.h = h;
                    self.h_computed = true;
                end
            else
                self.x = x;
                h = [-400*(x2-3*x1^2)+2 , -400*x1 ;
                     -400*x1            ,   200   ];
                self.h = h;
                self.h_computed = true;
                self.f_computed = false;
                self.g_computed = false;
            end
        end
        
        % Compute the product of the Hessian of f at x with vector x.
        function hv = hessvecprod(self, x, v)
            if length(x) ~= 2
                error('First input must be a 2 dimensional vector!');
            end
            if length(v) ~= 2
                error('Second input must be a 2 dimensional vector!');
            end
            x1 = x(1);
            x2 = x(2);
            if isequal( x, self.x )
                if self.h_computed
                    hv = self.h*v;
                else
                    self.h = [-400*(x2-3*x1^2)+2 , -400*x1 ;
                              -400*x1            ,   200   ]; 
                    self.h_computed = true;
                    hv = self.h*v;
                end
            else
                self.x = x;
                self.h = [-400*(x2-3*x1^2)+2 , -400*x1 ;
                          -400*x1            ,   200   ]; 
                self.h_computed = true;
                self.f_computed = false;
                self.g_computed = false;
                hv = self.h*v;
            end
        end
        
    end
end