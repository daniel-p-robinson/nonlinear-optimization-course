classdef Rosenbrock
    % This class defines the following Rosenbrock function:
    % f(x1,x2) = 100(x2 - x1^2)^2 + (1-x1)^2
    
    % List of properties needed for the class.
    properties 
        % No properties needed for the Rosenbrock function.
    end
    
    methods
        
        % Evaluate objective function f at x.
        function f = func(self, x)
            if size(x,1) ~= 2
                error([inputname(2), ' must be a 2 dimensional vector!']);
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