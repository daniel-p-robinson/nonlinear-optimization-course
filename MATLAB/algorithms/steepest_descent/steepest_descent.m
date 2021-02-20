function [x,info] = steepest_descent(f_hand,g_hand,x0,params)
%------------------------------------------------------------------------
% The function call
%      [x,info] = steepest_descent(f_hand,g_hand,x0,params )
% aims to compute a vector x that solves the optimization problem
%      minimize f(x)
% usign a steepest descent method.  The arguments are defined as follows:
%
%  Input arguments:
%  ----------------
%    f_hand     : handle of a function that computes f(x). The function
%                 funcf associated to the handle should be of the form
%                 [f] = funcf(x) where f = f(x) for a given input x.
%    g_hand     : handle to a function that computes grad f.  The function 
%                 funcg associated to the handle should be of the form
%                 [g] = funcg(x) where g = grad f(x) for a given input x.
%    x0         : initial guess of a minimizer of f.
%    params     : structure with the following members:
%                 maxiter    : maximum number of allowed iterations.
%                 printlevel : amount of printing to perform.
%                                 0  no printing
%                                 1  single line of output per iteration
%                              When printing, the following is displayed:
%                                 Iter       current iteration number
%                                 Norm-F     2-norm of F at current x
%                                 Norm-x     2-norm of current iterate x
%                                 Norm-step  2-norm of Newton step
%                 tol        : desired stoppping tolerance.
%                 outputfile : name of a file for output to be printed.
%                 probname   : problem name used for printing purposes.
%                 stepchoice : string indicating how choice of stepsize:
%                              fixed       fixed step size is used.
%                              adaptive    adaptive step size is used.
%                 stepsize   : a positive scalar value indicating the fixed
%                              step size to be used.  This value is only
%                              relevant when stepchoice = 'fixed' is used.
%  Output arguments:
%  -----------------
%    x      : final iterate computed
%    info   : structure with the following members:
%             f      : value of f at the final iterate
%             g      : value of the gradient of f at the final iterate
%             g_norm : norm of the gradient of f at the final iterate
%             iter   : total number of iterations performed
%             status : integer indicating the reason for termination
%                      0  Successful termination. A problem is considered 
%                         to have been successful solved if it finds an x
%                         satisfying ||g|| <= tol*max(1,||g0)||) whgere g
%                         is as above and g0 is the gradient of f evaluated
%                         at the input parameter x0.
%                      1  Step computed is too small to make more progress
%                      2  Maximum number of iterations is reached
%                     -1  An error in the inputs was detected
%                     -2  A NaN or Inf was detected
%                     -9  New warning encountered
%
% Author:
% ----------------
% Daniel P. Robinson
% Lehigh University
% Department of Industrial and Systems Engineering
% Bethlehem, PA, 18015, USA
%
% History:
% -----------------
% February 12, 2021:
%    - This is the original version.
%-------------------------------------------------------------------------

% Set dummy values for outputs; prevents errors resulting from bad inputs.  
x      = [];
f      = [];  info.f = f;
g      = [];  info.g = g;
g_norm = 0 ;  info.g_norm = g_norm;
iter   = 0 ;  info.iter = iter;
status = 0 ;  info.status = status;

% Check whether enough input parameters were supplied 
if nargin < 3
    fprintf('\n steepest_descent(ERROR):not enough input parameters.\n');
    info.status = -1;
    return
end

% If three input arguments, then define default values for fourth input.
if nargin == 3
    params.maxiter    = 500;
    params.printlevel = 1;
    params.tol        = 1e-6;
    params.outfileID  = 1;
    params.probname   = '';
    params.stepchoice = 'fixed';
    params.stepsize   = 1;
end

% Check that the initial point makes sense.
if length(x0) <= 0
    str = 'x0';
    fprintf('\n steepest_descent(ERROR):Invalid argument %s.\n',str);
    info.status = -1;
    return
end

% Check that all fields required in input params are supplied.
if isfield(params,'maxiter')
    maxiter = params.maxiter;
    if maxiter < 0
        str = 'MAXITER';
        fprintf('\n steepest_descent(ERROR):Invalid control parameter %s.\n',str);
        info.status = -1;
        return
    end
else
    str = 'MAXITER';
    fprintf('\n steepest_descent(ERROR):control parameter %s not supplied.\n',str);
    info.status = -1;
    return
end
    
if isfield(params,'printlevel')
    printlevel = params.printlevel;
    if printlevel < 0
        str = 'printlevel';
        fprintf('\n steepest_descent(ERROR):Invalid control parameter %s.\n',str);
        info.status = -1;
        return
    end
else
    str = 'PRINTLEVEL';
    fprintf('\n steepest_descent(ERROR):control parameter %s not supplied.\n',str);
    info.status = -1;
    return
end
    
if isfield(params,'tol')
    tol = params.tol;
    if tol < 0
        str = 'TOL';
        fprintf('\n steepest_descent(ERROR):Invalid control parameter %s.\n',str);
        info.status = -1;
        return
    end
else
    str = 'TOL';
    fprintf('\n steepest_descent(ERROR):control parameter %s not supplied.\n',str);
    info.status = -1;
    return
end

if isfield(params,'outfileID')
    outfileID = params.outfileID;
    if outfileID <= 0 || outfileID == 2
        str = 'outfileID';
        fprintf('\n steepest_descent(ERROR):Invalid control parameter %s.\n',str);
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
        fprintf('\n steepest_descen(ERROR):Invalid control parameter %s.\n',str);
        info.status = -1;
        return
    end
else
    probname = ''; 
end

if isfield(params,'stepchoice')
    stepchoice = params.stepchoice;
    if strcmp(stepchoice,'fixed')
        if isfield(params,'stepsize')
            stepsize = params.stepsize;
            if stepsize <= 0 || stepsize == inf
                str = 'STEPSIZE';
                fprintf('\n steepest_descent(ERROR):control parameter %s not a valid choice.\n',str);
                info.status = -1;
                return
            end
        else
            str = 'STEPSIZE';
            fprintf('\n steepest_descent(ERROR):control parameter %s not supplied.\n',str);
            info.status = -1;
            return
        end
    elseif strcmp(stepchoice,'adaptive')
        % Nothing to do.
    else
        str = 'STEPCHOICE';
        fprintf('\n steepest_descent(ERROR):control parameter %s not a valid choice.\n',str);
        info.status = -1;
        return
    end
else
    str = 'STEPCHOICE';
    fprintf('\n steepest_descent(ERROR):control parameter %s not supplied.\n',str);
    info.status = -1;
    return
end

% Check that inputs f_hand and g_hand are both function handles.
if ~isa(f_hand,'function_handle')
    str = 'f_hand';
    fprintf('\n steepest_descent(ERROR):Invalid control parameter %s.\n',str);
    info.status = -1;
    return
end

if ~isa(g_hand,'function_handle')
    str = 'g_hand';
    fprintf('\n steepest_descent(ERROR):Invalid control parameter %s.\n',str);
    info.status = -1;
    return
end

% Constant
TINY = eps^(2/3); % Determines if step is too small to make progress.

% Initialization.
x       = x0;
norm_x  = norm(x);
f       = feval(f_hand,x);
g       = feval(g_hand,x);
norm_g  = norm(g);
norm_g0 = norm_g; % Save value at x0 to use in relative stopping condition.

% Save things to make printing easier.
dashedline = repelem('-',1,81) ;
header     = '  Iter         f          norm_g         norm_x        stepsize       norm_step';

% Print column header and value of f at initial point.
if printlevel ~= 0  
  fprintf(outfileID,'%s\n',dashedline);
  fprintf(outfileID,'                            Steepest Descent Method \n');
  fprintf(outfileID,'%s\n',dashedline);
  fprintf(outfileID,' maximum iterations    : %g\n',maxiter);
  fprintf(outfileID,' print level           : %g\n',printlevel);
  fprintf(outfileID,' termination tolerance : %1.2e\n',tol);
  fprintf(outfileID,' file for output       : %s\n',outfileNAME);
  fprintf(outfileID,' problem name          : %s\n',probname);
  fprintf(outfileID,' step size choice      : %s\n',stepchoice);
  if strcmp(stepchoice,'fixed')
      fprintf(outfileID,' step size used        : %1.2e\n',stepsize);
  end
  fprintf(outfileID,'%s\n',dashedline);
  fprintf(outfileID,'%s\n',header);
  fprintf(outfileID,' %5g %14.7e %14.7e %14.7e', iter, f, norm_g, norm_x);
end

% Main loop: perform steepest descent iterations.
while ( 1 )

  % Check for termination
  if ( norm_g <= tol*max(1,norm_g0) )
     status = 0;
     outcome = ' Relative stopping tolerance reached';
     break
  elseif ( iter >= maxiter )
     status = 2;
     outcome = ' Maximum allowed iterations reached';
     break
  end
      
  % Choose the stepsize taken along the steepest descent direction.
  if strcmp(stepchoice,'fixed')
      alpha = stepsize;
  else
    fprintf('\n steepest_descent(ERROR): this part of code not implemented yet.\n');
    info.status = -1;
    return
  end
  
  % Step to actually take.
  p      = -alpha*g;
  norm_p = alpha*norm_g;
  
  % Save norm of current iterate before updating iterate.
  norm_xprev = norm_x;
  
  % Update the iterate and its associated values.
  iter   = iter + 1;
  x      = x + p;
  norm_x = norm(x);
  f      = feval(f_hand,x);
  g      = feval(g_hand,x);
  norm_g = norm(g);
  
  % Check for NaNs in F and J.
  if isnan(f) || isinf(f)
      status = -2;
      outcome = ' ERROR (NaN/Inf when evaluating f)';
      break
  elseif sum(isnan(g(:))) >= 1 || sum(isinf(g(:))) >= 1
      status = -2;
      outcome = ' ERROR (NaN/Inf when evaluating g)';
      break
  end
  
  % Print iterate information, if needed.
  if printlevel ~= 0
    fprintf(outfileID,' %14.7e %14.7e \n', stepsize, norm_p); % End previous line.
    if mod(iter,20) == 0
        fprintf(outfileID,'%s\n',header); % Print header every 20 iterations.
    end
    fprintf(outfileID,' %5g %14.7e %14.7e %14.7e', iter, f, norm_g, norm_x); % Start next line.
  end
  
  % Terminate if the step taken is too small to make significant progress.
  if norm_p < (1 + norm_xprev)*TINY
     status = 1;
     outcome = ' Step is too small to make additional progress.';
     break
  end
  
end

% Print termination message, if requested.
if printlevel
  fprintf(outfileID,'\n\n Result      :%s \n', outcome);
  if status == 1
    fprintf(outfileID,'              ||x||    : %13.7e\n', norm_x);
    fprintf(outfileID,'              ||step|| : %13.7e\n', norm_p);
  end
  fprintf(outfileID,' Iterations   : %-5g\n', iter);
  fprintf(outfileID,' Final f      : %13.7e\n', f );
  fprintf(outfileID,' Final ||g||  : %13.7e\n', norm_g );
  fprintf(outfileID,' Final ||g0|| : %13.7e\n', norm_g0 );  
  fprintf(outfileID,'%s\n',dashedline);
end

% Fill output variable inform.
info.f      = f;
info.g      = g;
info.norm_g = norm_g;
info.iter   = iter;
info.status = status;

return

end
