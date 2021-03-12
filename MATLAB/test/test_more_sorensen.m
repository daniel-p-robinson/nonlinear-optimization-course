% ------------------------------------------------------------------------
% Author  : Frank E. Curtis
% Purpose : Test the More-Sorensen solver on various trust region subproblems.
% Creation: First created on March 12, 2021.
% ------------------------------------------------------------------------

% Add path to functions and the solver.
addpath('../objective_functions/')
addpath('../algorithms/more_sorensen/')

% Save a dashed line for printing.
dashedline = repelem('-',1,90) ;

% ------------------------------------------
% Begin testing various subproblems.
% -------------------------------------------
fprintf('%s\n',dashedline)
fprintf(' Begin: Testing of More-Sorensen solver\n')
fprintf('%s\n',dashedline)

% Open a file for printing.
outfileID = fopen('test_more_sorensen.out','w+');

% Test: Generate random quadratic, and then run solver.
% -----------------------------------------------------------------------

num_instances = 9;

for instance = 1:num_instances
    
    % Gather the objective function for a Quadratic.
    fprintf(' Testing of More-Sorensen solver on QP-random(%g).......',instance)
    
    % Properties of the quadratic objective function to form.
    props.n       = 100;
    props.density = 0.2;
    props.rc      = 2^(-instance);
    props.kind    = 1;
    props.c_mean  = 1;
    props.c_sd    = 1;
    funobj        = Quadratic(props); %f(x) = g'x + 0.5x'Hx
    
    % Get g and H
    g = funobj.grad(zeros(props.n,1));
    H = funobj.hess(zeros(props.n,1));
    
    % Use More-Sorensen to solve trust region subproblem with radius 1.
    params.tol        = 1e-8;
    params.maxiter    = props.n;
    params.printlevel = 1;
    params.outfileID  = outfileID;
    params.probname   = sprintf('QP-random(%g)',instance);
    [~,info]          = more_sorensen(H,g,10^(ceil(num_instances/2)-instance),params);
    
    % Print the outcome.
    fprintf('exited with status = %2g\n',info.status);
    
end

% ------------
% Finish up.
% ------------
fprintf('%s\n',dashedline)
fprintf(' End: Testing of More-Sorensen solver\n')
fprintf('%s\n',dashedline)
fclose('all');
