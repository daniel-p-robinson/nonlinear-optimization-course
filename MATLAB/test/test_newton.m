% ------------------------------------------------------------------------
% Author: Daniel P. Robinson
% Purpose: Test the Newton's Method solver on various problems.
% Creation: First created on February 11, 2021.
% ------------------------------------------------------------------------

% Add path to the functions and Newton algorithm.
addpath('../objective_functions/')
addpath('../algorithms/newton/')

% Save a dashed line for printing.
dashedline = repelem('-',1,84) ;

% ------------------------------------------
% Begin testing various objective functions.
% -------------------------------------------
fprintf('%s\n',dashedline)
fprintf(' Begin: Testing of Newton solver\n')
fprintf('%s\n',dashedline)

% ------------------------------------------
% Test: Rosenbrock function
% ------------------------------------------

% Gather the object Rosenbrock.
fprintf(' Testing algorithm NEWTON on function Rosenbrock.............')
funobj = Rosenbrock;

% Define function handles for computing F and its Jacobian J.
Ffunc = @funobj.grad;
Jfunc = @funobj.hess;

% Initial estimate of a zero of F.
x0 = [10;20];

% Open a file for printing.
outfileID = fopen('test_newton.out','w+');

% Control parameters in structure params.
params.maxiter    = 30;            % Used for all problems.
params.printlevel = 1;             % Used for all problems.
params.tol        = 1e-9;          % Used for all problems.
params.outfileID  = outfileID;     % Used for all problems.
params.probname   = 'Rosenbrock';  % CHANGED for each problem below.

% Call Newton Method solver.
[~,info] = newton(Ffunc,Jfunc,x0,params);
fprintf('exited with status = %2g\n',info.status);

% ------------------------------------------
% Test: Genhumps function
% ------------------------------------------

% Gather the object Genhumps.
fprintf(' Testing algorithm NEWTON on function Genhumps...............')
funobj = Genhumps;

% Define function handles for computing F and its Jacobian J.
Ffunc = @funobj.grad;
Jfunc = @funobj.hess;

% Initial estimate of a zero of F.
x0 = ones(5,1);

% Name of function.
params.probname = 'Genhumps';

% Call Newton Method solver.
[~,info] = newton(Ffunc,Jfunc,x0,params);
fprintf('exited with status = %2g\n',info.status);

% ------------------------------------------
% Test: Quadratic function
% ------------------------------------------

% Gather the object Genhumps.
fprintf(' Testing algorithm NEWTON on function Quadratic..............')
props.n       = 100;
props.density = 0.2;
props.rc      = 1e-2;
props.kind    = 1;
props.c_mean  = 1;
props.c_sd    = 1;
funobj        = Quadratic(props);

% Define function handles for computing F and its Jacobian J.
Ffunc = @funobj.grad;
Jfunc = @funobj.hess;

% Initial estimate of a zero of F.
x0 = ones(funobj.n,1);

% Name of function.
params.probname = 'Quadratic (100,0.2,1e-3,1,1,1)';

% Call Newton Method solver.
[~,info] = newton(Ffunc,Jfunc,x0,params);
fprintf('exited with status = %2g\n',info.status);

% --------------------------------------------
% Test function: Least-Squares-Tukey function
% --------------------------------------------

% bodyfat data set
%------------------

% Gather the object LeastSquaresTukey.
fprintf(' Testing algorithm NEWTON on function LeastSquaresTukey......')
funobj = LeastSquaresTukey('../datasets/leastsquares/bodyfat.mat');

% Define function handles for computing F and its Jacobian J.
Ffunc = @funobj.grad;
Jfunc = @funobj.hess;

% Initial estimate of a zero of F.
x0 = zeros(size(funobj.A,2), 1);

% Name of function.
params.probname = 'Least-Squares-Tukey (data:bodyfat)';

% Call Newton Method solver.
[~,info] = newton(Ffunc,Jfunc,x0,params);
fprintf('exited with status = %2g\n',info.status);

% abalone data set
%------------------

% Gather the object LeastSquaresTukey.
fprintf(' Testing algorithm NEWTON on function LeastSquaresTukey......')
funobj = LeastSquaresTukey('../datasets/leastsquares/abalone.mat');

% Define function handles for computing F and its Jacobian J.
Ffunc = @funobj.grad;
Jfunc = @funobj.hess;

% Initial estimate of a zero of F.
x0 = zeros(size(funobj.A,2), 1);

% Name of function.
params.probname = 'Least-Squares-Tukey (data:abalone)';

% Call Newton Method solver.
[~,info] = newton(Ffunc,Jfunc,x0,params);
fprintf('exited with status = %2g\n',info.status);

% bodyfatExpand3 data set
% ------------------------

% Gather the object LeastSquaresTukey.
fprintf(' Testing algorithm NEWTON on function LeastSquaresTukey......')
funobj = LeastSquaresTukey('../datasets/leastsquares/bodyfatExpand3.mat');

% Define function handles for computing F and its Jacobian J.
Ffunc = @funobj.grad;
Jfunc = @funobj.hess;

% Initial estimate of a zero of F.
x0 = zeros(size(funobj.A,2), 1);

% Name of function.
params.probname = 'Least-Squares-Tukey (data:bodyfatExpand3)';

% Call Newton Method solver.
[~,info] = newton(Ffunc,Jfunc,x0,params);
fprintf('exited with status = %2g\n',info.status);

% ------------------------------------------
% Test function: Least-Squares
% ------------------------------------------

% bodyfat data set
%------------------

% Gather the object Least-Squares.
fprintf(' Testing algorithm NEWTON on function Least-Squares..........')
funobj = LeastSquares('../datasets/leastsquares/bodyfat.mat');

% Define function handles for computing F and its Jacobian J.
Ffunc = @funobj.grad;
Jfunc = @funobj.hess;

% Initial estimate of a zero of F.
x0 = zeros(size(funobj.A,2), 1);

% Name of function.
params.probname = 'Least-Squares (data:bodyfat)';

% Call Newton Method solver.
[~,info] = newton(Ffunc,Jfunc,x0,params);
fprintf('exited with status = %2g\n',info.status);

% abalone data set
%------------------

% Gather the object Least-Squares.
fprintf(' Testing algorithm NEWTON on function Least-Squares..........')
funobj = LeastSquares('../datasets/leastsquares/abalone.mat');

% Define function handles for computing F and its Jacobian J.
Ffunc = @funobj.grad;
Jfunc = @funobj.hess;

% Initial estimate of a zero of F.
x0 = zeros(size(funobj.A,2), 1);

% Name of function.
params.probname = 'Least-Squares (data:abalone)';

% Call Newton Method solver.
[~,info] = newton(Ffunc,Jfunc,x0,params);
fprintf('exited with status = %2g\n',info.status);

% bodyfatExpand3 data set
%------------------------

% Gather the object Least-Squares.
fprintf(' Testing algorithm NEWTON on function Least-Squares..........')
funobj = LeastSquares('../datasets/leastsquares/bodyfatExpand3.mat');

% Define function handles for computing F and its Jacobian J.
Ffunc = @funobj.grad;
Jfunc = @funobj.hess;

% Initial estimate of a zero of F.
x0 = zeros(size(funobj.A,2), 1);

% Name of function.
params.probname = 'Least-Squares (data:bodyfatExpand3)';

% Call Newton Method solver.
[~,info] = newton(Ffunc,Jfunc,x0,params);
fprintf('exited with status = %2g\n',info.status);

% ------------------------------------------
% Test function: Logistic Regression
% ------------------------------------------

% Diabetes data set
%------------------

% Gather the object Logistic
fprintf(' Testing algorithm NEWTON on function Logistic...............')
funobj = Logistic('../datasets/logistic/diabetes.mat');

% Define function handles for computing F and its Jacobian J.
Ffunc = @funobj.grad;
Jfunc = @funobj.hess;

% Initial estimate of a zero of F.
x0 = zeros(size(funobj.A,2), 1);

% Name of function.
params.probname = 'Logistic (data:diabetes)';

% Call Newton Method solver.
[~,info] = newton(Ffunc,Jfunc,x0,params);
fprintf('exited with status = %2g\n',info.status);

% Leu data set
%------------------

% Gather the object Logistic
fprintf(' Testing algorithm NEWTON on function Logistic...............')
funobj = Logistic('../datasets/logistic/leu.mat');

% Define function handles for computing F and its Jacobian J.
Ffunc = @funobj.grad;
Jfunc = @funobj.hess;

% Initial estimate of a zero of F.
x0 = zeros(size(funobj.A,2), 1);

% Name of function.
params.probname = 'Logistic (data:leu)';

% Call Newton Method solver.
[~,info] = newton(Ffunc,Jfunc,x0,params);
fprintf('exited with status = %2g\n',info.status);

% Phishing data set
%------------------

% Gather the object Logistic
fprintf(' Testing algorithm NEWTON on function Logistic...............')
funobj = Logistic('../datasets/logistic/phishing.mat');

% Define function handles for computing F and its Jacobian J.
Ffunc = @funobj.grad;
Jfunc = @funobj.hess;

% Initial estimate of a zero of F.
x0 = zeros(size(funobj.A,2), 1);

% Name of function.
params.probname = 'Logistic (data:phishing)';

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
