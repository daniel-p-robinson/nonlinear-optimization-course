function [x,info] = newton(Ffunc,Jfunc,x0,params)
%------------------------------------------------------------------------
%  This function is an implementation of Newton's Method. The function call 
%
%     [x,info] = newton( Ffunc,Jfunc,x0,params )
%
%  aims to compute a vector x such that F(x) = 0 for a function F(x). The
%  input and output arguments are describe next.
%
%  Input arguments:
%  ----------------
%    Ffunc      : handle of a function that computes F(x). The function
%                 specification should be of the form [F] = Ffunc(x). 
%    Jfunc      : handle to a function that computes J(x). he function
%                 specification should be of the form [J] = Jfunc(x).
%    x0         : initial guess at a zero of F. 
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
%
%  Output arguments:
%  -----------------
%    x      : final iterate computed
%    info   : structure with the following members:
%             F      : value of F at the final iterate
%             J      : value of the Jacobian of F at the final iterate
%             iter   : total number of iterations performed
%             status : integer indicating the reason for termination
%                      0  Successful termination. A problem is considered 
%                         to have been successful solved if it finds an x
%                         satisfying ||F(x)|| <= tol*max(1,||F(x_0)||).
%                      1  Newton step is too small to make further progress
%                      2  Maximum number of iterations was reached
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
% February 13, 2021:
%    - This is the original version.
%-------------------------------------------------------------------------

% Turn off certain warnings that I will explicitely handle in the code.
warning('off','MATLAB:illConditionedMatrix');
warning('off','MATLAB:singularMatrix');
warning('off','MATLAB:nearlySingularMatrix');

% Set dummy values for outputs; prevents errors resulting from bad inputs.  
x      = [];
F      = [];  info.F = F;
J      = [];  info.J = J;
iter   = 0 ;  info.iter = iter;
status = 0 ;  info.status = status;

% Check whether enough input parameters were supplied 
if nargin < 3
    fprintf('\n newton(ERROR):not enough input parameters supplied.\n');
    info.status = -1;
    return
end

% If three input arguments, then define default values for fourth input.
if nargin == 3
    params.maxiter    = 100;
    params.printlevel = 1;
    params.tol        = 1e-8;
    params.outfileID  = 1;
    params.probname   = ''; 
end

% Check that the initial point makes sense.
if length(x0) <= 0
    str = 'x0';
    fprintf('\n newton(ERROR):Invalid argument %s.\n',str);
    info.status = -1;
    return
end

% Check that all fields required in input params are supplied.
if isfield(params,'maxiter')
    maxiter = params.maxiter;
    if maxiter < 0
        str = 'MAXITER';
        fprintf('\n newton(ERROR):Invalid control parameter %s.\n',str);
        info.status = -1;
        return
    end
else
    str = 'MAXITER';
    fprintf('\n newton(ERROR):control parameter %s not supplied.\n',str);
    info.status = -1;
    return
end
    
if isfield(params,'printlevel')
    printlevel = params.printlevel;
    if printlevel < 0
        str = 'printlevel';
        fprintf('\n newton(ERROR):Invalid control parameter %s.\n',str);
        info.status = -1;
        return
    end
else
    str = 'PRINTLEVEL';
    fprintf('\n newton(ERROR):control parameter %s not supplied.\n',str);
    info.status = -1;
    return
end
    
if isfield(params,'tol')
    tol = params.tol;
    if tol < 0
        str = 'TOL';
        fprintf('\n newton(ERROR):Invalid control parameter %s.\n',str);
        info.status = -1;
        return
    end
else
    str = 'TOL';
    fprintf('\n newton(ERROR):control parameter %s not supplied.\n',str);
    info.status = -1;
    return
end

if isfield(params,'outfileID')
    outfileID = params.outfileID;
    if outfileID <= 0 || outfileID == 2
        str = 'outfileID';
        fprintf('\n newton(ERROR):Invalid control parameter %s.\n',str);
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
        fprintf('\n newton(ERROR):Invalid control parameter %s.\n',str);
        info.status = -1;
        return
    end
else
    probname = ''; 
end

% Check that inputs Ffunc and Jfunc are both function handles.
if ~isa(Ffunc,'function_handle')
    str = 'Ffunc';
    fprintf('\n newton(ERROR):Invalid control parameter %s.\n',str);
    info.status = -1;
    return
end

if ~isa(Jfunc,'function_handle')
    str = 'Jfunc';
    fprintf('\n newton(ERROR):Invalid control parameter %s.\n',str);
    info.status = -1;
    return
end

% Constant
TINY = eps^(2/3); % Determines if step is too small to make progress.

% Initialization.
x      = x0;
normx  = norm(x);
F      = feval(Ffunc,x);
J      = feval(Jfunc,x);
normF  = norm(F);
normF0 = normF;  % Save value at x0 for use in relative stopping condition.

% Save things to make printing easier.
dashedline = repelem('-',1,60) ;
header = '  Iter      Norm-F         Norm-x       Norm-step    Warning';

% Print column header and information about F at initial point.
if printlevel ~= 0  
  fprintf(outfileID,'%s\n',dashedline);
  fprintf(outfileID,'                     Newton''s Method \n');
  fprintf(outfileID,'%s\n',dashedline);
  fprintf(outfileID,' maximum iterations    : %g\n',maxiter);
  fprintf(outfileID,' print level           : %g\n',printlevel);
  fprintf(outfileID,' termination tolerance : %1.2e\n',tol);
  fprintf(outfileID,' file for output       : %s\n',outfileNAME);
  fprintf(outfileID,' problem name          : %s\n',probname);
  fprintf(outfileID,'%s\n',dashedline);
  fprintf(outfileID,'%s\n',header);
  fprintf(outfileID,' %5g %14.7e %14.7e', iter, normF, normx );
end

% Main loop: perform Newton iterations.
while ( 1 )

  % Check for termination
  if ( normF <= tol*max(1,normF0) )
     status = 0;
     outcome = ' Relative stopping tolerance reached';
     break
  elseif ( iter >= maxiter )
     status = 2;
     outcome = ' Maximum allowed iterations reached';
     break
  end
      
  % Compute Newton step p.
  % ----------------------
  lastwarn('', '');          % Reset the lastwarn message and id.
  p = - J\F;                 % Solve the Newton linear system.
  [~, warnId] = lastwarn();  % Check for warning.
  
  % Set warning string appropriately
  if( isempty(warnId) )
      warnstring = '   -   ';
  elseif strcmp(warnId,'MATLAB:singularMatrix')
      warnstring = '  sing ';
  elseif strcmp(warnId,'MATLAB:illConditionedMatrix')
      warnstring = 'ill-cond';
  elseif strcmp(warnId,'MATLAB:nearlySingularMatrix')
      warnstring = 'nearsing';
  else
      status = -9;
      outcome = ' ERROR (unknown NEW warning encountered)';
      break
  end
  
  % Set norm of the Newton step.
  if sum(isnan(p)) >= 1 || sum(isinf(p)) >= 1
      status = -2;
      outcome = ' ERROR (NaN/Inf in the search direction)';
      break
  else
      normp = norm(p);
  end
  
  % Save norm of current iterate before moving updating.
  normxprev = normx;
  
  % Update the iterate and its associated values.
  iter  = iter + 1;
  x     = x + p;
  normx = norm(x);
  F     = feval(Ffunc,x);
  J     = feval(Jfunc,x);
  normF = norm(F);
  
  % Check for NaNs in F and J.
  if sum(isnan(F)) >= 1 || sum(isinf(F)) >= 1
      status = -2;
      outcome = ' ERROR (NaN when evaluating F)';
      break
  elseif sum(isnan(J(:))) >= 1 || sum(isinf(J(:))) >= 1
      status = -2;
      outcome = ' ERROR (NaN when evaluating J)';
      break
  end
  
  % Print iterate information, if needed.
  if printlevel ~= 0
    fprintf(outfileID,' %14.7e  %s \n', normp, warnstring);      % End previous line. 
    fprintf(outfileID,' %5g %14.7e %14.7e', iter, normF, normx); % Start next line.
  end
  
  % Terminate if Newton step was too small to make significant progress.
  if normp < (1 + normxprev)*TINY
     status = 1;
     outcome = ' Newton step is too small to make additional progress.';
     break
  end
  
end

% Print termination message, if requested.
if printlevel
  fprintf(outfileID,'\n\n Result     :%s \n', outcome);
  if status == 1
    fprintf(outfileID,'              ||x||    : %13.7e\n', normx);
    fprintf(outfileID,'              ||step|| : %13.7e\n', normp);
  end
  fprintf(outfileID,' Iterations : %-5g\n', iter);
  fprintf(outfileID,' Final |F|  : %13.7e\n', normF );
  fprintf(outfileID,'%s\n',dashedline);
end

% Fill output variable inform.
info.F      = F;
info.normF  = normF;
info.J      = J;
info.iter   = iter;
info.status = status;

return

end
