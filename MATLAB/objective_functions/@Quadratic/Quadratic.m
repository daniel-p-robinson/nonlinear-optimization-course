classdef Quadratic < handle
% =========================================================================
% This class defines a quadratic objective function of the form
%       f(x) = 1/2 x'Ax + c
% with the variables x being an n-dimensional vector. There are three ways
% of creating an instance of this class. We discuss these three now.
% 1. The first allowed call to create an instance of this class is
%       Quadratic(struct)
% which will create a randomly generated quadratic objective function based 
% on the values included in the input structure struct.  The members are:
%       struct.n        length of x
%       struct.density  density of matrix A
%       struct.rc       the reciprical of the condition number of A
%       struct.kind     has the value either 0 or 1.  This is used in the
%                       call to the Matlab built-in function sprandsym, so
%                       see the help for that function for details.
%       struct.c_mean   mean for the distribution from which the entries
%                       for c will be selected.
%       struct.c_sd     standard deviation for distribution from which the
%                       entries for c will be selected
% 2. The second allowed call to create an instance of this class is
%       Quadratic(path_to_mat_file)
% where path_to_mat_file is a string holding the full path to a .mat file,
% from which the matrix A and vector c will be loaded.
% 3. The third allowed call to create an instance of this class is
%       Quadratic(A,c)
% where A is a symmetric matrix and c a vector with dimension matching A.
%=========================================================================
    % List of properties needed for the class.
    properties
         A         % matrix in the quadratic function
         c         % linear term in the quadratic function
         n         % number of optimization variables
         x         % saved vector x
         Ax        % product A*x
         f_computed % 0/1 indicates whether f(x) has been computed
         f          % holds f(x) when f_computed is true
         g_computed % 0/1 indicates whether grad f(x) has been computed
         g          % holds grad f(x) when g_computed is true
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
                    elseif ~isfield(varargin{1},'c_mean')
                        error('Quadratic(ERROR): structure field (c_mean) is missing.')
                    elseif ~isfield(varargin{1},'c_sd')
                        error('Quadratic(ERROR): structure field (c_sd) is missing.')    
                    else                        
                       n       = varargin{1}.n;
                       density = varargin{1}.density;
                       rc      = varargin{1}.rc;
                       kind    = varargin{1}.kind;
                       self.A  = sprandsym(n,density,rc,kind);
                       c_mean  = varargin{1}.c_mean;
                       c_sd    = varargin{1}.c_sd;
                       self.c  = c_mean + c_sd.*randn(n,1);
                    end
                elseif isfile([varargin{1},'.mat'])
                    load([varargin{1},'.mat'],'A','c');
                    self.A = A;
                    self.c = c;
                elseif isfile(varargin{1})
                    load(varargin{1},'A','c');
                    self.A = A;
                    self.c = c;
                else
                    error('Quadratic(ERROR): invalid input argument.')
                end
             elseif nargin == 2
                self.A = varargin{1};
                self.c = varargin{2};
             else
                error('Quadratic(ERROR): wrong number of inputs.')
            end
            if ~isreal(self.A) || ~issymmetric(self.A)
                error('Quadratic(ERROR): A must be real and symmetric.')
            end
            self.n    = size(self.A,2);
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
            self.x    = zeros(self.n,1);
            self.Ax   = zeros(self.n,1);
            self.f_computed = false;  self.f = [];
            self.g_computed = false;  self.g = [];
            self.name = 'Quadratic';
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
                    f = (self.c)'*x + (1/2)*x'*self.Ax;
                    self.f = f;
                    self.f_computed = true;
                end
            else
                self.x = x;
                self.Ax = self.A * self.x;
                f = (self.c)'*x + (1/2)*x'*self.Ax;
                self.f = f;   
                self.f_computed = true;
                self.g_computed = false;
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
                    g = self.c + self.Ax;
                    self.g = g;
                    self.g_computed = true;
                end
            else
                self.x = x;
                self.Ax = self.A * self.x;
                g = self.c + self.Ax;
                self.g = g;
                self.g_computed = true;
                self.f_computed = false;
            end
        end
        
        % Evaluate the Hessian of f at x.
        function h = hess(self, x)
            if length(x) ~= self.n
                error('Input must be a %g dimensional vector!',self.n);
            end
            h = self.A;
        end
        
        % Compute the product of the Hessian of f at x with vector x.
        function hv = hessvecprod(self, x, v)
            if length(x) ~= self.n
                error('First input must be a %g dimensional vector!',self.n);
            end
            if length(v) ~= self.n
                error('Second input must be a %g dimensional vector!',self.n);
            end
            hv = (self.A)*v;
        end
        
    end
end