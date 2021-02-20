classdef Quadratic
% =========================================================================
% This class defines a quadratic objective function.
%=========================================================================
    % List of properties needed for the class.
    properties
         H         % matrix in the quadratic function
         g         % linear term in the quadratic function
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
        function self=Quadratic(varargin)
            if nargin < 1
               error('Quadratic(ERROR): no inputs supplied.') 
            elseif nargin == 1
                if isstruct(varargin{1})
                    if ~isfield(varargin{1},'n')
                        error('Quadratic(ERROR): structure field (n) is missing.')
                    elseif ~isfield(varargin{1},'density')
                        error('Quadratic(ERROR): structure field (density) is missing.')
                    elseif ~isfield(varargin{1},'rc')
                        error('Quadratic(ERROR): structure field (rc) is missing.')
                    elseif ~isfield(varargin{1},'kind')
                        error('Quadratic(ERROR): structure field (kind) is missing.')
                    elseif ~isfield(varargin{1},'g_mean')
                        error('Quadratic(ERROR): structure field (g_mean) is missing.')
                    elseif ~isfield(varargin{1},'g_sd')
                        error('Quadratic(ERROR): structure field (g_sd) is missing.')    
                    else                        
                       n       = varargin{1}.n;
                       density = varargin{1}.density;
                       rc      = varargin{1}.rc;
                       kind    = varargin{1}.kind;
                       self.H  = sprandsym(n,density,rc,kind);
                       g_mean  = varargin{1}.g_mean;
                       g_sd    = varargin{1}.g_sd;
                       self.g  = g_mean + g_sd.*randn(n,1);
                    end
                elseif isfile([varargin{1},'.mat'])
                    load([varargin{1},'.mat'],'H','g');
                    self.H = H;
                    self.g = g;
                elseif isfile(varargin{1})
                    load(varargin{1},'H','g');
                    self.H = H;
                    self.g = g;
                else
                    error('Quadratic(ERROR): invalid input argument.')
                end
             elseif nargin == 2
                self.H = varargin{1};
                self.g = varargin{2};
             else
                error('Quadratic(ERROR): wrong number of inputs.')
            end
            self.n    = size(self.H,2);
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
            self.name      = 'Quadratic';
        end
        
        % Evaluate objective function f at x.
        function f = func(self, x)
            if length(x) ~= self.n  
                error([inputname(2), sprintf(' must be a %g dimensional vector!',self.n)]);
            end
            f =  (self.g)'*x + (1/2)*x'*(self.H)*x;
        end
        
        % Evaluate the gradient of f at x.            
        function g = grad(self, x)
            if length(x) ~= self.n
                error([inputname(2), ' must be a %g dimensional vector!',self.n]);
            end
            g = self.g + (self.H)*x;
        end
        
        % Evaluate the Hessian of f at x.
        function h = hess(self, x)
            if length(x) ~= self.n
                error([inputname(2), ' must be a %g dimensional vector!',self.n]);
            end
            h = self.H;
        end
        
        % Compute the product of the Hessian of f at x with vector x.
        function hv = hessvecprod(self, x, v)
            if length(x) ~= self.n
                error([inputname(2), ' must be a %g dimensional vector!',self.n]);
            end
            if length(v) ~= self.n
                error([inputname(3), ' must be a %g dimensional vector!',self.n]);
            end
            hv     = (self.H)*v;
        end
        
    end
end