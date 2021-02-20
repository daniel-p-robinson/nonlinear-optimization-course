classdef Genhumps
% =========================================================================
% This class defines the objective function for a 5-dimensional problem
% with a lot of humps. This problem is from the well-known CUTEr test set.
%=========================================================================
    % List of properties needed for the class.
    properties 
         n         % number of optimization variables
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
        function self=Genhumps(varargin)
            self.n   = 5;
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
            self.name      = 'Genhumps';
        end
        
        % Evaluate objective function f at x.
        function f = func(self, x)
            if length(x) ~= 5
                error([inputname(2), sprintf(' must be a %g dimensional vector!',5)]);
            end
            f = 0;
            for i = 1:4
                f = f + sin(2*x(i))^2*sin(2*x(i+1))^2 + 0.05*(x(i)^2 + x(i+1)^2);
            end
        end
        
        % Evaluate the gradient of f at x.            
        function g = grad(self, x)
            if length(x) ~= 5
                error([inputname(2), ' must be a %g dimensional vector!',5]);
            end
            g = [4*sin(2*x(1))*cos(2*x(1))* sin(2*x(2))^2                  + 0.1*x(1);
                 4*sin(2*x(2))*cos(2*x(2))*(sin(2*x(1))^2 + sin(2*x(3))^2) + 0.2*x(2);
                 4*sin(2*x(3))*cos(2*x(3))*(sin(2*x(2))^2 + sin(2*x(4))^2) + 0.2*x(3);
                 4*sin(2*x(4))*cos(2*x(4))*(sin(2*x(3))^2 + sin(2*x(5))^2) + 0.2*x(4);
                 4*sin(2*x(5))*cos(2*x(5))* sin(2*x(4))^2                  + 0.1*x(5)];
        end
        
        % Evaluate the Hessian of f at x.
        function h = hess(self, x)
            if length(x) ~= 5
                error([inputname(2), ' must be a %g dimensional vector!',5]);
            end
            h      = zeros(5,5);
            h(1,1) =  8* sin(2*x(2))^2*(cos(2*x(1))^2 - sin(2*x(1))^2) + 0.1;
            h(1,2) = 16* sin(2*x(1))*cos(2*x(1))*sin(2*x(2))*cos(2*x(2));
            h(2,2) =  8*(sin(2*x(1))^2 + sin(2*x(3))^2)*(cos(2*x(2))^2 - sin(2*x(2))^2) + 0.2;
            h(2,3) = 16* sin(2*x(2))*cos(2*x(2))*sin(2*x(3))*cos(2*x(3));
            h(3,3) =  8*(sin(2*x(2))^2 + sin(2*x(4))^2)*(cos(2*x(3))^2 - sin(2*x(3))^2) + 0.2;
            h(3,4) = 16* sin(2*x(3))*cos(2*x(3))*sin(2*x(4))*cos(2*x(4));
            h(4,4) =  8*(sin(2*x(3))^2 + sin(2*x(5))^2)*(cos(2*x(4))^2 - sin(2*x(4))^2) + 0.2;
            h(4,5) = 16* sin(2*x(4))*cos(2*x(4))*sin(2*x(5))*cos(2*x(5));
            h(5,5) =  8* sin(2*x(4))^2*(cos(2*x(5))^2 - sin(2*x(5))^2) + 0.1;
            h(2,1) = h(1,2);
            h(3,2) = h(2,3);
            h(4,3) = h(3,4);
            h(5,4) = h(4,5);
        end
        
        % Compute the product of the Hessian of f at x with vector x.
        function hv = hessvecprod(self, x, v)
            if length(x) ~= 5
                error([inputname(2), ' must be a %g dimensional vector!',5]);
            end
            if length(v) ~= 5
                error([inputname(3), ' must be a %g dimensional vector!',5]);
            end
            h      = zeros(5,5);
            h(1,1) =  8* sin(2*x(2))^2*(cos(2*x(1))^2 - sin(2*x(1))^2) + 0.1;
            h(1,2) = 16* sin(2*x(1))*cos(2*x(1))*sin(2*x(2))*cos(2*x(2));
            h(2,2) =  8*(sin(2*x(1))^2 + sin(2*x(3))^2)*(cos(2*x(2))^2 - sin(2*x(2))^2) + 0.2;
            h(2,3) = 16* sin(2*x(2))*cos(2*x(2))*sin(2*x(3))*cos(2*x(3));
            h(3,3) =  8*(sin(2*x(2))^2 + sin(2*x(4))^2)*(cos(2*x(3))^2 - sin(2*x(3))^2) + 0.2;
            h(3,4) = 16* sin(2*x(3))*cos(2*x(3))*sin(2*x(4))*cos(2*x(4));
            h(4,4) =  8*(sin(2*x(3))^2 + sin(2*x(5))^2)*(cos(2*x(4))^2 - sin(2*x(4))^2) + 0.2;
            h(4,5) = 16* sin(2*x(4))*cos(2*x(4))*sin(2*x(5))*cos(2*x(5));
            h(5,5) =  8* sin(2*x(4))^2*(cos(2*x(5))^2 - sin(2*x(5))^2) + 0.1;
            h(2,1) = h(1,2);
            h(3,2) = h(2,3);
            h(4,3) = h(3,4);
            h(5,4) = h(4,5);
            hv     = h*v;
        end
        
    end
end