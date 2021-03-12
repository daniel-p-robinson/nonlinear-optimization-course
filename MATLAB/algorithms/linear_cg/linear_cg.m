function [x,info] = linear_cg(Av_hand,b,x,params)
% ========================================================================
% This function implements the linear conjugate-gradient algorithm, which
% is an iterative algorithm for approximately solving a linear system of
% the form Ax = b where the matrix A is nxn, symmetric, and positive
% definite. A valid call for using this function is of the form
%      [x,info] = linear_cg(Av_hand,b,x,params)
% where Av_hand is a handle that computes the product of the matrix A with
% a desired input vector v, b is a vector of length n, x is an initial
% guess of the unique solution of the linear system Ax = b, and params is a
% structure that may contain input parameters (see below).
% ========================================================================
% Author      : Frank E. Curtis and Daniel P. Robinson
% Description : Linear conjugate gradient method
% Input       : Av_hand ~ handle of a function that computes A*v for a 
%                         desired vector v.  The function, say Av_prod,
%                         associated to the handle should be of the form
%                            [Av] = Av_prod(v)
%                         where the output Av = A*v for the given input v.
%               b       ~ right-hand side vector
%               x       ~ initial iterate
%               params  ~ structure with the following members:
%                         tol         solution tolerance
%                         maxiter     maximum allowed number of iterations
%                         printlevel  integer indicating print details
%                                     0  no printing will be performed
%                                     1  single line output per iteration
%                         outfileNAME name of a file for printing output.
%                         probname    string holding the problem name.
% Output      : x    ~ final iterate
%             : info ~ structure whose members include the following:
%                      status  integer indicates the outcome:
%                              0  final termination residual was reached
%                              1  maximum number of iterations was reached
%                             -1  invalid parameter was provided
%                      outcome string given the reason for termination
%                      k       number of iterations performed
%                      r       residual vector of the linear system at x
%                      r_norm  norm of the residual vector r
% ========================================================================

% Make sure enough input parameters are supplied.
if nargin < 4
    info.status = -1;
    fprintf('\n cg(ERROR):not enough input parameters.\n');
    return
end

% Make sure first input is a function handle
if ~isa(Av_hand,'function_handle')
    str = 'Av_hand';
    fprintf('\n cg(ERROR):Invalid input parameter %s.\n',str);
    info.status = -1;
    return
end

% Make sure that inputs b and x are valid.
m = length(b);
if m <= 0
    str = 'b';
    fprintf('\n cg(ERROR):Invalid input parameter %s.\n',str);
    info.status = -1;
    return
end
n = length(x);
if n <= 0
    str = 'x';
    fprintf('\n cg(ERROR):Invalid input parameter %s.\n',str);
    info.status = -1;
    return
end

% Make sure the final input, which holds the control parameters, is okay.
% If any of the required fields are not supplied, set default values.
if ~isa(params,'struct')
    str = 'params';
    fprintf('\n cg(ERROR):Invalid input parameter %s.\n',str);
    info.status = -1;
    return
end

str2 = 'params';

if isfield(params,'tol')
    tol = params.tol;
    if any(tol < 0)
        str1 = 'tol';
        fprintf('\n cg(ERROR):Invalid field %s in input %s.\n',str1,str2);
        info.status = -1;
        return
    end
else
    tol = 1e-6;
end

if isfield(params,'maxiter')
    maxiter = params.maxiter;
    if maxiter < 0
        str1 = 'maxiter';
        fprintf('\n cg(ERROR):Invalid field %s in input %s.\n',str1,str2);
        info.status = -1;
        return
    end
else
    maxiter = n;
end

if isfield(params,'printlevel')
    printlevel = params.printlevel;
else
    printlevel = 1; % Default is to print output to screen.
end

if isfield(params,'outfileID')
    outfileID = params.outfileID;
    if outfileID <= 0 || outfileID == 2
        str = 'outfileID';
        fprintf('\n cg(ERROR):Invalid control parameter %s.\n',str);
        info.status = -1;
        return
    end
else
    outfileID = 1; % standard output (the screen)
end
outfileNAME = fopen(outfileID);

if isfield(params,'probname')
    probname = params.probname;
    if ~ischar(probname)
        str = 'probname';
        fprintf('\n cgf(ERROR):Invalid control parameter %s.\n',str);
        info.status = -1;
        return
    end
else
    probname = ''; 
end

% Store output strings
out_line = '====================================================================';
out_data = '  k       ||x||       ||r||   |    ||p||        alpha        beta';
out_null =                                 '----------  -----------  -----------';

% Print output header
if printlevel >= 1
    fprintf(outfileID,'%s\n',out_line);
    fprintf(outfileID,'                 Linear Conjugate Gradient Method\n');
    fprintf(outfileID,'%s\n',out_line);
    fprintf(outfileID,' maximum iterations    : %g\n',maxiter);
    fprintf(outfileID,' print level           : %g\n',printlevel);
    fprintf(outfileID,' file for output       : %s\n',outfileNAME);
    fprintf(outfileID,' problem name          : %s\n',probname);
    fprintf(outfileID,'%s\n%s\n%s\n',out_line,out_data,out_line);
    
end

% Evaluate initial iterate norm
norms.x = norm(x);

% Evaluate initial residual
Ax = feval(Av_hand,x);
r = Ax-b;

% Evaluate residual norm
norms.r = norm(r);

% Store initial residual norm
norms.r0 = norms.r;

% Evaluate initial direction
p = -r;

% Evaluate direction norm
norms.p = norm(p);

% Initialize iteration counter
k = 0;

% Main CG loop
while (1)  
    
  % Print iterate information
  if printlevel >= 1
      fprintf(outfileID,'%5d  %.4e  %.4e | ',k,norms.x,norms.r);
  end
  
  % Check for termination
  if norms.r <= max(norms.r0,1)*tol
      outcome = 'Termination tolerance met.';
      status = 0;
      break
  elseif k >= maxiter
      outcome = 'Maximum number of iterations reached.';
      status = 1;
      break
  end
  
  % Evaluate matrix-vector product
  Ap = feval(Av_hand,p);
    
  % Evaluate vector-vector product
  pAp = p'*Ap;
  
  % Evaluate steplength
  alpha = (norms.r)^2/pAp;
  
  % Update iterate
  x = x + alpha*p;
  
  % Evaluate iterate norm
  norms.x = norm(x);
  
  % Before updating the residual, save the current residual norm.
  norms.r_prev = norms.r;
    
  % Update residual
  r = r + alpha*Ap;
  
  % Evaluate residual norm
  norms.r = norm(r);
  
  % Evaluate CG multiplier
  beta = (norms.r/norms.r_prev)^2;
  
  % Print step information
  if printlevel >= 1
      fprintf(outfileID,'%.4e  %+.4e  %+.4e\n',norms.p,alpha,beta);
  end
  
  % Update direction
  p = -r + beta*p;
  
  % Evaluate direction norm
  norms.p = norm(p);
  
  % Increment iteration counter
  k = k + 1;
  
  % Print output header
  if printlevel >= 1
      if mod(k,20) == 0
          fprintf(outfileID,'%s\n%s\n%s\n',out_line,out_data,out_line);
      end
  end
  
end

% Print output footer
fprintf(outfileID,'%s\n%s\n',out_null,out_line);
fprintf(outfileID,'Number of iterations  : %g\n', k);
fprintf(outfileID,'Status on termination : %-2g\n', status);
fprintf(outfileID,'Outcome               : %s\n',outcome);
fprintf(outfileID,'Final residual norm   : %13.7e\n', norms.r); 
fprintf(outfileID,'%s\n',out_line);

% Fill output parameters.
info.k       = k;
info.status  = status;
info.outcome = outcome;
info.r       = r;
info.r_norm  = norms.r;

end