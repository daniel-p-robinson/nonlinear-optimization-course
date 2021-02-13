% ------------------------------------------------------------------------
% Author: Daniel P. Robinson
% Purpose: Test the Newton's Method solver on various problems.
% History:
%   February 11, 2021: Original version.
% ------------------------------------------------------------------------

% Add path to the functions and Newton algorithm.
addpath('/Users/danielrobinson/daniel/git/nonlinear-optimization-course/MATLAB/objective_functions/')
addpath('/Users/danielrobinson/daniel/git/nonlinear-optimization-course/MATLAB/algorithms/newton/')

% Save a dashed line for printing.
dashedline = repelem('-',1,77) ;

% ------------------------------------------
% Begin testing various objective functions.
% -------------------------------------------
fprintf('%s\n',dashedline)
fprintf(' Begin: Testing of Newton solver\n')
fprintf('%s\n',dashedline)

% Test 1: Rosenbrock function
% ------------------------------------------

% Gather the object Rosenbrock.
fprintf(' Testing algorithm NEWTON on function Rosenbrock......')
funobj = Rosenbrock;

% Define function handles for computing F and its Jacobian J.
Ffunc = @funobj.grad;
Jfunc = @funobj.hess;

% Initial estimate of a zero of F.
x0 = [10;20];

% Open a file for printing.
outfileID = fopen('test_newton.out','w+');

% Control parameters in structure params.
params.maxiter    = 30;
params.printlevel = 1;
params.tol        = 1e-9;
params.outfileID  = outfileID;

% Call Newton Method solver.
[~,info] = newton(Ffunc,Jfunc,x0,params);
fprintf('exited with status = %2g\n',info.status);

% Test 2: Least-Squares
% ------------------------------------------

% Gather the object Least-Squares.
fprintf(' Testing algorithm NEWTON on function Least-Squares...')
funobj = LeastSquares('/Users/danielrobinson/daniel/git/nonlinear-optimization-course/MATLAB/datasets/leastsquares/bodyfat.mat');

% Define function handles for computing F and its Jacobian J.
Ffunc = @funobj.grad;
Jfunc = @funobj.hess;

% Initial estimate of a zero of F.
x0 = ones(size(funobj.A,2), 1);

% Call Newton Method solver.
[~,info] = newton(Ffunc,Jfunc,x0,params);
fprintf('exited with status = %2g\n',info.status);

% Test 3: Logistic Regression
% ------------------------------------------

% Gather the object Logistic
fprintf(' Testing algorithm NEWTON on function Logistic........')
funobj = Logistic('/Users/danielrobinson/daniel/git/nonlinear-optimization-course/MATLAB/datasets/logistic/diabetes.mat');

% Define function handles for computing F and its Jacobian J.
Ffunc = @funobj.grad;
Jfunc = @funobj.hess;

% Initial estimate of a zero of F.
x0 = ones(size(funobj.A,2), 1);

% Call Newton Method solver.
[~,info] = newton(Ffunc,Jfunc,x0,params);
fprintf('exited with status = %2g\n',info.status);

% ------------
% Finish up.
% ------------
fprintf('%s\n',dashedline)
fprintf(' End: Testing of NEWTON solver\n')
fprintf('%s\n',dashedline)
fclose('all');
