% ------------------------------------------------------------------------
% Author: Daniel P. Robinson
% Purpose: Test the Newton's Method solver on various problems.
% History:
%   February 11, 2021: Original version.
% ------------------------------------------------------------------------

% Add path to the functions and Newton algorithm.
addpath('../objective_functions/')
addpath('../algorithms/newton/')

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

% bodyfat data set
%------------------

% Gather the object Least-Squares.
fprintf(' Testing algorithm NEWTON on function Least-Squares...')
funobj = LeastSquares('../datasets/leastsquares/bodyfat.mat');

% Define function handles for computing F and its Jacobian J.
Ffunc = @funobj.grad;
Jfunc = @funobj.hess;

% Initial estimate of a zero of F.
x0 = zeros(size(funobj.A,2), 1);

% Call Newton Method solver.
[~,info] = newton(Ffunc,Jfunc,x0,params);
fprintf('exited with status = %2g\n',info.status);

% abalone data set
%------------------

% Gather the object Least-Squares.
fprintf(' Testing algorithm NEWTON on function Least-Squares...')
funobj = LeastSquares('../datasets/leastsquares/abalone.mat');

% Define function handles for computing F and its Jacobian J.
Ffunc = @funobj.grad;
Jfunc = @funobj.hess;

% Initial estimate of a zero of F.
x0 = zeros(size(funobj.A,2), 1);

% Call Newton Method solver.
[~,info] = newton(Ffunc,Jfunc,x0,params);
fprintf('exited with status = %2g\n',info.status);

% bodyfatExpand3 data set
%------------------------

% Gather the object Least-Squares.
fprintf(' Testing algorithm NEWTON on function Least-Squares...')
funobj = LeastSquares('../datasets/leastsquares/bodyfatExpand3.mat');

% Define function handles for computing F and its Jacobian J.
Ffunc = @funobj.grad;
Jfunc = @funobj.hess;

% Initial estimate of a zero of F.
x0 = zeros(size(funobj.A,2), 1);

% Call Newton Method solver.
[~,info] = newton(Ffunc,Jfunc,x0,params);
fprintf('exited with status = %2g\n',info.status);

% Test 3: Logistic Regression
% ------------------------------------------

% Diabetes data set
%------------------

% Gather the object Logistic
fprintf(' Testing algorithm NEWTON on function Logistic........')
funobj = Logistic('../datasets/logistic/diabetes.mat');

% Define function handles for computing F and its Jacobian J.
Ffunc = @funobj.grad;
Jfunc = @funobj.hess;

% Initial estimate of a zero of F.
x0 = zeros(size(funobj.A,2), 1);

% Call Newton Method solver.
[~,info] = newton(Ffunc,Jfunc,x0,params);
fprintf('exited with status = %2g\n',info.status);

% Leu data set
%------------------

% Gather the object Logistic
fprintf(' Testing algorithm NEWTON on function Logistic........')
funobj = Logistic('../datasets/logistic/leu.mat');

% Define function handles for computing F and its Jacobian J.
Ffunc = @funobj.grad;
Jfunc = @funobj.hess;

% Initial estimate of a zero of F.
x0 = zeros(size(funobj.A,2), 1);

% Call Newton Method solver.
[~,info] = newton(Ffunc,Jfunc,x0,params);
fprintf('exited with status = %2g\n',info.status);

% Phishing data set
%------------------

% Gather the object Logistic
fprintf(' Testing algorithm NEWTON on function Logistic........')
funobj = Logistic('../datasets/logistic/phishing.mat');

% Define function handles for computing F and its Jacobian J.
Ffunc = @funobj.grad;
Jfunc = @funobj.hess;

% Initial estimate of a zero of F.
x0 = zeros(size(funobj.A,2), 1);

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
