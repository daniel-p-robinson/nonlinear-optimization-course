classdef Rosenbrock
    % This class defines the following Rosenbrock function:
    % f(x1,x2) = 100(x2 - x1^2)^2 + (1-x1)^2
    
    % List of properties needed for the class.
    properties 
        n         % number of optimization variables
        mE        % number of equality constraints: c_E(x) = 0
        mI        % number of inequality constraints: cl <= c_I(x) <= cu
        cL        % lower bound vector for inequality constraints
        cU        % upper bound vector for inequality constraints
        xL        % lower bound vector for optimization variables: xL <= x
        xU        % upper bound vector for optimization variables: x <= xU
        x0        % initial solution estimate
        yE0       % initial Lagrange multiplier for cE(x) = 0.
        yI0       % initial Lagrange multiplier for cL <=  cI(x) <= cU.
        obj_type  % type of objective function:     linear, quadratic, linear-least-squares, nonlinear-least-squares, general
        conE_type % type of equality constraints:   linear, quadratic, general
        conI_type % type of inequality constraints: linear, quadratic, general
        name      % name associated to the optimization problem               
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
        end
        
        % Evaluate objective function f at x.
        function f = func(self, x)
            if length(x) ~= self.n
                error([inputname(2), sprintf(' must be a %g dimensional vector!',self.n)]);
            end
            x1 = x(1);
            x2 = x(2);
            f = 100*(x2-x1^2)^2 + (1-x1)^2;
        end
        
        % Evaluate the gradient of f at x.            
        function g = grad(self, x)
            if size(x,1) ~= 2
                error([inputname(2), ' must be a 2 dimensional vector!']);
            end
            x1 = x(1);
            x2 = x(2);
            g = [-400*x1*(x2-x1^2) - 2*(1-x1) ;
                     200*(x2-x1^2)            ];
        end
        
        % Evaluate the Hessian of f at x.
        function h = hess(self, x)
            if size(x,1) ~= 2
                error([inputname(2), ' must be a 2 dimensional vector!']);
            end
            x1 = x(1);
            x2 = x(2);
            h = [-400*(x2-3*x1^2)+2 , -400*x1 ;
                 -400*x1            ,   200   ];
        end
        
        % Compute the product of the Hessian of f at x with vector x.
        function hv = hessvecprod(self, x, v)
            if size(x,1) ~= 2
                error([inputname(2), ' must be a 2 dimensional vector!']);
            end
            if size(v,1) ~= 2
                error([inputname(3), ' must be a 2 dimensional vector!']);
            end
            x1 = x(1);
            x2 = x(2);
            h = [-400*(x2-3*x1^2)+2 , -400*x1 ;
                 -400*x1            ,   200   ];
            hv = h*v;
        end
        
    end
end