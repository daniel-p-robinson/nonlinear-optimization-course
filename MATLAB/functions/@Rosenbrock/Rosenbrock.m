classdef Rosenbrock
    % this class defines the Rosenbrock function
    % f(x) = 100(x_2 - x_1^2)^2 + (1-x_1)^2
    properties    
    end
    
    methods
        function f = func(self, x)
            if size(x,1) ~= 2
                error([inputname(2), ' must be a 2 dimensional vector!']);
            end
            f = 100*(x(2)-x(1)^2)^2+(1-x(1))^2;
        end
        
        function g = grad(self, x)
            if size(x,1) ~= 2
                error([inputname(2), ' must be a 2 dimensional vector!']);
            end
            g = [-x(1)*400*(x(2)-x(1)^2)-2*(1-x(1));
                     200*(x(2)-x(1)^2)            ];
        end
        
        function h = hess(self, x)
            if size(x,1) ~= 2
                error([inputname(2), ' must be a 2 dimensional vector!']);
            end
            h = [-400*(x(2)-3*x(1)^2)+2 -400*x(1);
                 -400*x(1)               200      ];
        end
        
        function hv = hvprod(self, x, v)
            if size(x,1) ~= 2
                error([inputname(2), ' must be a 2 dimensional vector!']);
            end
            if size(x,1) ~= 2
                error([inputname(2), ' must be a 2 dimensional vector!']);
            end
            h = [-400*(x(2)-3*x(1)^2)+2 -400*x(1);
                 -400*x(1)               200      ];
            hv = h * v;
        end
    end
end


% function v = rosenbrock(x,o)
% 
% % function v = rosenbrock(x,o)
% %
% % Author      : Frank E. Curtis
% % Description : Rosenbrock function evaluator.
% % Input       : x ~ current iterate
% %               o ~ evaluation option
% %                     0 ~ function value
% %                     1 ~ gradient value
% %                     2 ~ Hessian value
% % Output      : v ~ function, gradient, or Hessian value
% 
% % Switch on o
% switch o
%   
%   case 0
% 
%     % Evaluate function
%     v = 100*(x(2)-x(1)^2)^2+(1-x(1))^2;
%   
%   case 1
%   
%     % Evaluate gradient
%     v = [-x(1)*400*(x(2)-x(1)^2)-2*(1-x(1));
%                200*(x(2)-x(1)^2)            ];
% 
%   case 2
% 
%     % Evaluate Hessian
%     v = [-400*(x(2)-3*x(1)^2)+2 -400*x(1);
%          -400*x(1)               200      ];
%   
% end