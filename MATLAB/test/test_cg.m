% ------------------------------------------------------------------------
% Author: Daniel P. Robinson
% Purpose: Test the Conjugqter Gradient solver on various linear systems.
% Creation: First created on March 10, 2021.
% ------------------------------------------------------------------------

% Add path to the functions and Newton algorithm.
addpath('../objective_functions/')
addpath('../algorithms/linear_cg/')

% Save a dashed line for printing.
dashedline = repelem('-',1,90) ;

% ------------------------------------------
% Begin testing various linear systems.
% -------------------------------------------
fprintf('%s\n',dashedline)
fprintf(' Begin: Testing of Linear Conjugate-Gradient solver\n')
fprintf('%s\n',dashedline)

% Open a file for printing.
outfileID = fopen('test_cg.out','w+');

% Test: Generate random quadratic, and then run CG on the linear system.
% -----------------------------------------------------------------------

num_instances = 9;

for instance = 1:num_instances
    
    % Gather the object function for the QuadraticRosenbrock.
    fprintf(' Testing algorithm CG on QP-random(%g).......',instance)
    
    % Properties of the quadratic objective function to form.
    props.n       = 100;
    props.density = 0.2;
    props.rc      = 1e-2;
    props.kind    = 1;
    props.c_mean  = 1;
    props.c_sd    = 1;
    funobj        = Quadratic(props); %f(x) = c'x + 0.5x'Ax
    
    % Can get c by evaluating the grad of f at 0.
    c = funobj.grad(zeros(props.n,1));
    
    % Define the function that will compute A*v for given input v.
    dummy = zeros(props.n,1);
    hvfunc_onlyv = @(v)funobj.hessvecprod(dummy,v); 

    % Use CG to solve, i.e., to solve Ax = -c.
    b                 = -c;
    x                 = zeros(props.n,1);
    params.e          = 1e-6*ones(length(b),1);
    params.maxiter    = props.n;
    params.printlevel = 1;
    params.outfileID  = outfileID;
    params.probname   = sprintf('Random QP(%g)',instance);
    [~,info]          = cg(hvfunc_onlyv,b,x,params);
    
    % Print the outcome.
    fprintf('exited with status = %2g\n',info.status);
    
end

% Test: Generate quadratics with increasing numbers of clusters of
% eignevalues.  Then check if CG takes that many iterations.
% -----------------------------------------------------------------------

num_instances = 9;

for instance = 1:num_instances
    
    % Gather the object function for the QuadraticRosenbrock.
    fprintf(' Testing algorithm CG on QP-clustered(%g)....',instance)
    
    % Properties of the quadratic objective function to form.
    n       = max(100,num_instances);
    density = 0.2;
    rc      = 1e-2;
    kind    = 1;
    b_mean  = 1;
    b_sd    = 1;
    
    A     = sprandsym(n,density,rc,kind);
    A     = full(A);
    [V,D] = eig(A);
    Dmod = ones(n,1);
    if instance >= 2
        Dmod(1:instance-1) = rand(instance-1,1);
    end
    A = V*diag(Dmod)*V';
    b  = b_mean + b_sd.*randn(n,1);
    
    % Define the function that will compute A*v for given input v.
    Av_hand = @(v)(A*v); 

    % Use CG to solve Ax = b
    x                 = zeros(n,1);
    params.e          = 1e-6*ones(length(b),1);
    params.maxiter    = n;
    params.printlevel = 1;
    params.outfileID  = outfileID;
    params.probname   = sprintf('Random QP with clustered eigs(%g)',instance);
    [~,info]          = cg(Av_hand,b,x,params);
    
    % Print the outcome.
    fprintf('exited with status = %2g....in....%2g iterations\n',info.status,info.k);
    
end

% ------------
% Finish up.
% ------------
fprintf('%s\n',dashedline)
fprintf(' End: Testing of Linear Conjugate-Gradient solver\n')
fprintf('%s\n',dashedline)
fclose('all');










% % ------------------------------------------
% % Test: Genhumps function
% % ------------------------------------------
% 
% % Gather the object Genhumps.
% fprintf(' Testing algorithm NEWTON on function Genhumps...............')
% funobj = Genhumps;
% 
% % Define function handles for computing F and its Jacobian J.
% Ffunc = @funobj.grad;
% Jfunc = @funobj.hess;
% 
% % Initial estimate of a zero of F.
% x0 = ones(5,1);
% 
% % Name of function.
% params.probname = 'Genhumps';
% 
% % Call Newton Method solver.
% [~,info] = newton(Ffunc,Jfunc,x0,params);
% fprintf('exited with status = %2g\n',info.status);
% 
% % ------------------------------------------
% % Test: Quadratic function
% % ------------------------------------------
% 
% % Gather the object Genhumps.
% fprintf(' Testing algorithm NEWTON on function Quadratic..............')
% props.n       = 100;
% props.density = 0.2;
% props.rc      = 1e-2;
% props.kind    = 1;
% props.c_mean  = 1;
% props.c_sd    = 1;
% funobj        = Quadratic(props);
% 
% % Define function handles for computing F and its Jacobian J.
% Ffunc = @funobj.grad;
% Jfunc = @funobj.hess;
% 
% % Initial estimate of a zero of F.
% x0 = ones(funobj.n,1);
% 
% % Name of function.
% params.probname = 'Quadratic (100,0.2,1e-3,1,1,1)';
% 
% % Call Newton Method solver.
% [~,info] = newton(Ffunc,Jfunc,x0,params);
% fprintf('exited with status = %2g\n',info.status);
% 
% % --------------------------------------------
% % Test function: Least-Squares-Tukey function
% % --------------------------------------------
% 
% % bodyfat data set
% %------------------
% 
% % Gather the object LeastSquaresTukey.
% fprintf(' Testing algorithm NEWTON on function LeastSquaresTukey......')
% funobj = LeastSquaresTukey('../datasets/leastsquares/bodyfat.mat');
% 
% % Define function handles for computing F and its Jacobian J.
% Ffunc = @funobj.grad;
% Jfunc = @funobj.hess;
% 
% % Initial estimate of a zero of F.
% x0 = zeros(size(funobj.A,2), 1);
% 
% % Name of function.
% params.probname = 'Least-Squares-Tukey (data:bodyfat)';
% 
% % Call Newton Method solver.
% [~,info] = newton(Ffunc,Jfunc,x0,params);
% fprintf('exited with status = %2g\n',info.status);
% 
% % abalone data set
% %------------------
% 
% % Gather the object LeastSquaresTukey.
% fprintf(' Testing algorithm NEWTON on function LeastSquaresTukey......')
% funobj = LeastSquaresTukey('../datasets/leastsquares/abalone.mat');
% 
% % Define function handles for computing F and its Jacobian J.
% Ffunc = @funobj.grad;
% Jfunc = @funobj.hess;
% 
% % Initial estimate of a zero of F.
% x0 = zeros(size(funobj.A,2), 1);
% 
% % Name of function.
% params.probname = 'Least-Squares-Tukey (data:abalone)';
% 
% % Call Newton Method solver.
% [~,info] = newton(Ffunc,Jfunc,x0,params);
% fprintf('exited with status = %2g\n',info.status);
% 
% % bodyfatExpand3 data set
% % ------------------------
% 
% % Gather the object LeastSquaresTukey.
% fprintf(' Testing algorithm NEWTON on function LeastSquaresTukey......')
% funobj = LeastSquaresTukey('../datasets/leastsquares/bodyfatExpand3.mat');
% 
% % Define function handles for computing F and its Jacobian J.
% Ffunc = @funobj.grad;
% Jfunc = @funobj.hess;
% 
% % Initial estimate of a zero of F.
% x0 = zeros(size(funobj.A,2), 1);
% 
% % Name of function.
% params.probname = 'Least-Squares-Tukey (data:bodyfatExpand3)';
% 
% % Call Newton Method solver.
% [~,info] = newton(Ffunc,Jfunc,x0,params);
% fprintf('exited with status = %2g\n',info.status);
% 
% % ------------------------------------------
% % Test function: Least-Squares
% % ------------------------------------------
% 
% % bodyfat data set
% %------------------
% 
% % Gather the object Least-Squares.
% fprintf(' Testing algorithm NEWTON on function Least-Squares..........')
% funobj = LeastSquares('../datasets/leastsquares/bodyfat.mat');
% 
% % Define function handles for computing F and its Jacobian J.
% Ffunc = @funobj.grad;
% Jfunc = @funobj.hess;
% 
% % Initial estimate of a zero of F.
% x0 = zeros(size(funobj.A,2), 1);
% 
% % Name of function.
% params.probname = 'Least-Squares (data:bodyfat)';
% 
% % Call Newton Method solver.
% [~,info] = newton(Ffunc,Jfunc,x0,params);
% fprintf('exited with status = %2g\n',info.status);
% 
% % abalone data set
% %------------------
% 
% % Gather the object Least-Squares.
% fprintf(' Testing algorithm NEWTON on function Least-Squares..........')
% funobj = LeastSquares('../datasets/leastsquares/abalone.mat');
% 
% % Define function handles for computing F and its Jacobian J.
% Ffunc = @funobj.grad;
% Jfunc = @funobj.hess;
% 
% % Initial estimate of a zero of F.
% x0 = zeros(size(funobj.A,2), 1);
% 
% % Name of function.
% params.probname = 'Least-Squares (data:abalone)';
% 
% % Call Newton Method solver.
% [~,info] = newton(Ffunc,Jfunc,x0,params);
% fprintf('exited with status = %2g\n',info.status);
% 
% % bodyfatExpand3 data set
% %------------------------
% 
% % Gather the object Least-Squares.
% fprintf(' Testing algorithm NEWTON on function Least-Squares..........')
% funobj = LeastSquares('../datasets/leastsquares/bodyfatExpand3.mat');
% 
% % Define function handles for computing F and its Jacobian J.
% Ffunc = @funobj.grad;
% Jfunc = @funobj.hess;
% 
% % Initial estimate of a zero of F.
% x0 = zeros(size(funobj.A,2), 1);
% 
% % Name of function.
% params.probname = 'Least-Squares (data:bodyfatExpand3)';
% 
% % Call Newton Method solver.
% [~,info] = newton(Ffunc,Jfunc,x0,params);
% fprintf('exited with status = %2g\n',info.status);
% 
% % ------------------------------------------
% % Test function: Logistic Regression
% % ------------------------------------------
% 
% % Diabetes data set
% %------------------
% 
% % Gather the object Logistic
% fprintf(' Testing algorithm NEWTON on function Logistic...............')
% funobj = Logistic('../datasets/logistic/diabetes.mat');
% 
% % Define function handles for computing F and its Jacobian J.
% Ffunc = @funobj.grad;
% Jfunc = @funobj.hess;
% 
% % Initial estimate of a zero of F.
% x0 = zeros(size(funobj.A,2), 1);
% 
% % Name of function.
% params.probname = 'Logistic (data:diabetes)';
% 
% % Call Newton Method solver.
% [~,info] = newton(Ffunc,Jfunc,x0,params);
% fprintf('exited with status = %2g\n',info.status);
% 
% % Leu data set
% %------------------
% 
% % Gather the object Logistic
% fprintf(' Testing algorithm NEWTON on function Logistic...............')
% funobj = Logistic('../datasets/logistic/leu.mat');
% 
% % Define function handles for computing F and its Jacobian J.
% Ffunc = @funobj.grad;
% Jfunc = @funobj.hess;
% 
% % Initial estimate of a zero of F.
% x0 = zeros(size(funobj.A,2), 1);
% 
% % Name of function.
% params.probname = 'Logistic (data:leu)';
% 
% % Call Newton Method solver.
% [~,info] = newton(Ffunc,Jfunc,x0,params);
% fprintf('exited with status = %2g\n',info.status);
% 
% % Phishing data set
% %------------------
% 
% % Gather the object Logistic
% fprintf(' Testing algorithm NEWTON on function Logistic...............')
% funobj = Logistic('../datasets/logistic/phishing.mat');
% 
% % Define function handles for computing F and its Jacobian J.
% Ffunc = @funobj.grad;
% Jfunc = @funobj.hess;
% 
% % Initial estimate of a zero of F.
% x0 = zeros(size(funobj.A,2), 1);
% 
% % Name of function.
% params.probname = 'Logistic (data:phishing)';
% 
% % Call Newton Method solver.
% [~,info] = newton(Ffunc,Jfunc,x0,params);
% fprintf('exited with status = %2g\n',info.status);
% 
% % ------------
% % Finish up.
% % ------------
% fprintf('%s\n',dashedline)
% fprintf(' End: Testing of NEWTON solver\n')
% fprintf('%s\n',dashedline)
% fclose('all');
