% ------------------------------------------------------------------------
% Author: Daniel P. Robinson
% Purpose: Test steepest descent method solver on various problems.
% First created on February 12, 2021.
% ------------------------------------------------------------------------

% Add path to the functions and Newton algorithm.
addpath('../objective_functions/')
addpath('../algorithms/steepest_descent/')

% Save a dashed line for printing.
dashedline = repelem('-',1,93);

% ------------------------------------------
% Begin testing various objective functions.
% -------------------------------------------
fprintf('%s\n',dashedline)
fprintf(' Begin: Testing of steepest descent solver\n')
fprintf('%s\n',dashedline)

% ------------------------------------------
% Test function: Rosenbrock function
% ------------------------------------------

% Gather the object Rosenbrock.
fprintf(' Testing algorithm STEEPEST_DESCENT on function Rosenbrock............')
funobj = Rosenbrock;

% Define function handles for computing F and its Jacobian J.
f_hand = @funobj.func;
g_hand = @funobj.grad;

% Initial estimate of a minimizer of f.
x0 = [10;20];

% Open a file for printing.
outfileID = fopen('test_steepest_descent.out','w+');

% Control parameters in structure params.
params.maxiter    = 1e+4;         % Used for all problems.
params.printlevel = 1;            % Used for all problems.
params.tol        = 1e-4;         % Used for all problems.
params.stepchoice = 'fixed';      % Used for all problems.
params.stepsize   = 1e-1;         % Used for all problems.
params.outfileID  = outfileID;    % Used for all problems.
params.probname   = 'Rosenbrock'; % CHANGES for each new problem below.

% Call steepest descent solver.
[~,info] = steepest_descent(f_hand,g_hand,x0,params);
fprintf('exited with status = %2g\n',info.status);

% ------------------------------------------
% Test: Genhumps function
% ------------------------------------------

% Gather the object Genhumps.
fprintf(' Testing algorithm STEEPEST_DESCENT on function Genhumps..............')
funobj = Genhumps;

% Define function handles for computing F and its Jacobian J.
f_hand = @funobj.func;
g_hand = @funobj.grad;

% Initial estimate of a zero of F.
x0 = ones(5,1);

% Name of function.
params.probname = 'Genhumps';

% Call steepest descent solver.
[~,info] = steepest_descent(f_hand,g_hand,x0,params);
fprintf('exited with status = %2g\n',info.status);

% ------------------------------------------
% Test: Quadratic function
% ------------------------------------------

% Gather the object Quadratic.
fprintf(' Testing algorithm STEEPEST_DESCENT on function Quadratic.............')
props.n       = 100;
props.density = 0.2;
props.rc      = 1e-2;
props.kind    = 1;
props.g_mean  = 1;
props.g_sd    = 1;
funobj        = Quadratic(props);

% Define function handles for computing F and its Jacobian J.
f_hand = @funobj.func;
g_hand = @funobj.grad;

% Initial estimate of a zero of F.
x0 = ones(funobj.n,1);

% Name of function.
params.probname = 'Quadratic (100,0.2,1e-3,1,1,1)';

% Call steepest descent solver.
[~,info] = steepest_descent(f_hand,g_hand,x0,params);
fprintf('exited with status = %2g\n',info.status);

% ------------------------------------------
% Test function: Least-Squares-Tukey
% ------------------------------------------

% bodyfat data set
% -----------------

% Gather the object Least-Squares-Tukey.
fprintf(' Testing algorithm STEEPEST_DESCENT on function Least-Squares-Tukey...')
funobj = LeastSquaresTukey('../datasets/leastsquares/bodyfat.mat');

% Define function handles for computing F and its Jacobian J.
f_hand = @funobj.func;
g_hand = @funobj.grad;

% Initial estimate of a minmizer of f.
x0 = zeros(size(funobj.A,2), 1);

% Name of function.
params.probname = 'Least-Squares-Tukey (data:bodyfat)';

% Call steepest descent solver.
[~,info] = steepest_descent(f_hand,g_hand,x0,params);
fprintf('exited with status = %2g\n',info.status);

% abalone data set
% -----------------

% Gather the object Least-Squares-Tukey.
fprintf(' Testing algorithm STEEPEST_DESCENT on function Least-Squares-Tukey...')
funobj = LeastSquares('../datasets/leastsquares/abalone.mat');

% Define function handles for computing F and its Jacobian J.
f_hand = @funobj.func;
g_hand = @funobj.grad;

% Initial estimate of a minmizer of f.
x0 = zeros(size(funobj.A,2), 1);

% Name of function.
params.probname = 'Least-Squares-Tukey (data:abalone)';

% Call steepest descent solver.
[~,info] = steepest_descent(f_hand,g_hand,x0,params);
fprintf('exited with status = %2g\n',info.status);

% bodyfatExpand3 data set
% ------------------------

% Gather the object Least-Squares-Tukey.
fprintf(' Testing algorithm STEEPEST_DESCENT on function Least-Squares-Tukey...')
funobj = LeastSquares('../datasets/leastsquares/bodyfatExpand3.mat');

% Define function handles for computing F and its Jacobian J.
f_hand = @funobj.func;
g_hand = @funobj.grad;

% Initial estimate of a minmizer of f.
x0 = zeros(size(funobj.A,2), 1);

% Name of function.
params.probname = 'Least-Squares-Tukey (data:bodyfatExpand3)';

% Call steepest descent solver.
[~,info] = steepest_descent(f_hand,g_hand,x0,params);
fprintf('exited with status = %2g\n',info.status);

% ------------------------------------------
% Test function: Least-Squares
% ------------------------------------------

% bodyfat data set
% -----------------

% Gather the object Least-Squares.
fprintf(' Testing algorithm STEEPEST_DESCENT on function Least-Squares.........')
funobj = LeastSquares('../datasets/leastsquares/bodyfat.mat');

% Define function handles for computing F and its Jacobian J.
f_hand = @funobj.func;
g_hand = @funobj.grad;

% Initial estimate of a minmizer of f.
x0 = zeros(size(funobj.A,2), 1);

% Name of function.
params.probname = 'Least-Squares (data:bodyfat)';

% Call steepest descent solver.
[~,info] = steepest_descent(f_hand,g_hand,x0,params);
fprintf('exited with status = %2g\n',info.status);

% abalone data set
% -----------------

% Gather the object Least-Squares.
fprintf(' Testing algorithm STEEPEST_DESCENT on function Least-Squares.........')
funobj = LeastSquares('../datasets/leastsquares/abalone.mat');

% Define function handles for computing F and its Jacobian J.
f_hand = @funobj.func;
g_hand = @funobj.grad;

% Initial estimate of a minmizer of f.
x0 = zeros(size(funobj.A,2), 1);

% Name of function.
params.probname = 'Least-Squares (data:abalone)';

% Call steepest descent solver.
[~,info] = steepest_descent(f_hand,g_hand,x0,params);
fprintf('exited with status = %2g\n',info.status);

% bodyfatExpand3 data set
% ------------------------

% Gather the object Least-Squares.
fprintf(' Testing algorithm STEEPEST_DESCENT on function Least-Squares.........')
funobj = LeastSquares('../datasets/leastsquares/bodyfatExpand3.mat');

% Define function handles for computing F and its Jacobian J.
f_hand = @funobj.func;
g_hand = @funobj.grad;

% Initial estimate of a minmizer of f.
x0 = zeros(size(funobj.A,2), 1);

% Name of function.
params.probname = 'Least-Squares (data:bodyfatExpand3)';

% Call steepest descent solver.
[~,info] = steepest_descent(f_hand,g_hand,x0,params);
fprintf('exited with status = %2g\n',info.status);

% ------------------------------------------
% Test function: Logistic Regression
% ------------------------------------------

% diabetes data set
%-------------------

% Gather the object Logistic
fprintf(' Testing algorithm STEEPEST_DESCENT on function Logistic..............')
funobj = Logistic('../datasets/logistic/diabetes.mat');

% Define function handles for computing F and its Jacobian J.
f_hand = @funobj.func;
g_hand = @funobj.grad;

% Initial estimate of a minimize of f.
x0 = ones(size(funobj.A,2), 1);

% Name of function.
params.probname = 'Logistic (data:diabetes)';

% Call steepest descent solver.
[~,info] = steepest_descent(f_hand,g_hand,x0,params);
fprintf('exited with status = %2g\n',info.status);

% leu data set
%-------------------

% Gather the object Logistic
fprintf(' Testing algorithm STEEPEST_DESCENT on function Logistic..............')
funobj = Logistic('../datasets/logistic/leu.mat');

% Define function handles for computing F and its Jacobian J.
f_hand = @funobj.func;
g_hand = @funobj.grad;

% Initial estimate of a minimize of f.
x0 = zeros(size(funobj.A,2), 1);

% Name of function.
params.probname = 'Logistic (data:leu)';

% Call steepest descent solver.
[~,info] = steepest_descent(f_hand,g_hand,x0,params);
fprintf('exited with status = %2g\n',info.status);

% phishing data set
%-------------------

% Gather the object Logistic
fprintf(' Testing algorithm STEEPEST_DESCENT on function Logistic..............')
funobj = Logistic('../datasets/logistic/phishing.mat');

% Define function handles for computing F and its Jacobian J.
f_hand = @funobj.func;
g_hand = @funobj.grad;

% Initial estimate of a minimize of f.
x0 = zeros(size(funobj.A,2), 1);

% Name of function.
params.probname = 'Logistic (data:phishing)';

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
