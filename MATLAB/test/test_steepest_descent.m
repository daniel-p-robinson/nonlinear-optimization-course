% ------------------------------------------------------------------------
% Author: Daniel P. Robinson
% Purpose: Test steepest descent method solver on various problems.
% History:
%   February 12, 2021: Original version.
% ------------------------------------------------------------------------

% Add path to the functions and Newton algorithm.
addpath('../objective_functions/')
addpath('../algorithms/steepest_descent/')

% Save a dashed line for printing.
dashedline = repelem('-',1,87) ;

% ------------------------------------------
% Begin testing various objective functions.
% -------------------------------------------
fprintf('%s\n',dashedline)
fprintf(' Begin: Testing of steepest descent solver\n')
fprintf('%s\n',dashedline)

% Test 1: Rosenbrock function
% ------------------------------------------

% Gather the object Rosenbrock.
fprintf(' Testing algorithm STEEPEST_DESCENT on function Rosenbrock......')
funobj = Rosenbrock;

% Define function handles for computing F and its Jacobian J.
f_hand = @funobj.func;
g_hand = @funobj.grad;

% Initial estimate of a minimizer of f.
x0 = [10;20];

% Open a file for printing.
outfileID = fopen('test_steepest_descent.out','w+');

% Control parameters in structure params.
params.maxiter    = 10000;
params.printlevel = 1;
params.tol        = 1e-4;
params.stepchoice = 'fixed';
params.stepsize   = 1e-1;
params.outfileID  = outfileID;

% Call steepst descent solver.
[~,info] = steepest_descent(f_hand,g_hand,x0,params);
fprintf('exited with status = %2g\n',info.status);

% Test 2: Least-Squares
% ------------------------------------------

% bodyfat data set
% -----------------

% Gather the object Least-Squares.
fprintf(' Testing algorithm STEEPEST_DESCENT on function Least-Squares...')
funobj = LeastSquares('../datasets/leastsquares/bodyfat.mat');

% Define function handles for computing F and its Jacobian J.
f_hand = @funobj.func;
g_hand = @funobj.grad;

% Initial estimate of a minmizer of f.
x0 = zeros(size(funobj.A,2), 1);

% Call steepest descent solver.
[~,info] = steepest_descent(f_hand,g_hand,x0,params);
fprintf('exited with status = %2g\n',info.status);

% abalone data set
% -----------------

% Gather the object Least-Squares.
fprintf(' Testing algorithm STEEPEST_DESCENT on function Least-Squares...')
funobj = LeastSquares('../datasets/leastsquares/abalone.mat');

% Define function handles for computing F and its Jacobian J.
f_hand = @funobj.func;
g_hand = @funobj.grad;

% Initial estimate of a minmizer of f.
x0 = zeros(size(funobj.A,2), 1);

% Call steepest descent solver.
[~,info] = steepest_descent(f_hand,g_hand,x0,params);
fprintf('exited with status = %2g\n',info.status);

% bodyfatExpand3 data set
% ------------------------

% Gather the object Least-Squares.
fprintf(' Testing algorithm STEEPEST_DESCENT on function Least-Squares...')
funobj = LeastSquares('../datasets/leastsquares/bodyfatExpand3.mat');

% Define function handles for computing F and its Jacobian J.
f_hand = @funobj.func;
g_hand = @funobj.grad;

% Initial estimate of a minmizer of f.
x0 = zeros(size(funobj.A,2), 1);

% Call steepest descent solver.
[~,info] = steepest_descent(f_hand,g_hand,x0,params);
fprintf('exited with status = %2g\n',info.status);

% Test 3: Logistic Regression
% ------------------------------------------

% diabetes data set
%-------------------

% Gather the object Logistic
fprintf(' Testing algorithm STEEPEST_DESCENT on function Logistic........')
funobj = Logistic('../datasets/logistic/diabetes.mat');

% Define function handles for computing F and its Jacobian J.
f_hand = @funobj.func;
g_hand = @funobj.grad;

% Initial estimate of a minimize of f.
x0 = ones(size(funobj.A,2), 1);

% Call steepest descent solver.
[~,info] = steepest_descent(f_hand,g_hand,x0,params);
fprintf('exited with status = %2g\n',info.status);

% leu data set
%-------------------

% Gather the object Logistic
fprintf(' Testing algorithm STEEPEST_DESCENT on function Logistic........')
funobj = Logistic('../datasets/logistic/leu.mat');

% Define function handles for computing F and its Jacobian J.
f_hand = @funobj.func;
g_hand = @funobj.grad;

% Initial estimate of a minimize of f.
x0 = zeros(size(funobj.A,2), 1);

% Call steepest descent solver.
[~,info] = steepest_descent(f_hand,g_hand,x0,params);
fprintf('exited with status = %2g\n',info.status);

% phishing data set
%-------------------

% Gather the object Logistic
fprintf(' Testing algorithm STEEPEST_DESCENT on function Logistic........')
funobj = Logistic('../datasets/logistic/phishing.mat');

% Define function handles for computing F and its Jacobian J.
f_hand = @funobj.func;
g_hand = @funobj.grad;

% Initial estimate of a minimize of f.
x0 = zeros(size(funobj.A,2), 1);

% Call steepest descent solver.
[~,info] = steepest_descent(f_hand,g_hand,x0,params);
fprintf('exited with status = %2g\n',info.status);

% ------------
% Finish up.
% ------------
fprintf('%s\n',dashedline)
fprintf(' End: Testing of STEEPEST_DESCENT solver\n')
fprintf('%s\n',dashedline)
fclose('all');
