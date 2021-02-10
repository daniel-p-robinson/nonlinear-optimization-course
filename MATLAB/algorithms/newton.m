function [x,F,J,iter,status] = newton(funobj,x0,maxiter,printlevel,tol)
%------------------------------------------------------------------------
%  The function call
%
%     [x,F,J,iter,stauts] = newton( Fname,x0,maxiter,printlevel,tol )
%
%  aims to compute a vector x such that F(x) = 0 for a function F(x). 
%  The zero is sought by using Newton's method.
%
%  Input arguments:
%  ----------------
%
%    Fname      : string containing the name of an m-file that 
%                 evaluates a function F and its Jaobian. The function
%                 value and Jacobian of F must be defined in an M-file
%                 named Fname with specification [F,J] = Fname(x). 
%    x0         : initial guess at a zero of F. 
%    maxiter    : maximum number of allowed iterations.
%    printlevel : amount of printing to perform.
%                    0  no printing
%                    1  single line of output per iteration
%                 When printing is requested, the following is displayed:
%                    Iter       current iteration number
%                    Norm-F     two-norm of F at the current iterate x
%                    Norm-x     two-norm of current iterate x
%                    Norm-step  two-norm of Newton step during iteration
%    tol        : desired stoppping tolerance on the size of F(x).
%
%  Output arguments:
%  -----------------
%
%    x      : final iterate computed
%    F      : value of F at the final computed iterate
%    J      : value of the Jacobian of F at the final computed iterate
%    iter   : total number of iterations performed
%    status : integer indicating the reason for termination
%              0  Successful termination sine norm(F(x)) <= tol
%              1  Newton step is too small to make any further progress
%              2  Maximum number of iterations was reached
%             -1  An error in the inputs was detected.
%
% Author:
% ----------------
% Daniel P. Robinson
% Lehigh University
% Department of Industrial and Systems Engineering
% Bethlehem, PA, 18015, USA
%
% This version is dated March 3, 2020.
%-------------------------------------------------------------------------

% Set dummy values for output arguments.
% Will prevent errors if termination because of bad input arguments.  
x = [];
F = [];
J = [];
iter = 0;
status = 0;

% Mark sure the correct number of arguments are passed in.
if nargin < 5
    fprintf('newton(ERROR):Wrong number of input arguments.\n');
    status = -1; 
    return
end

% Check to make sure that the arguments passed in make sense.
% if ~(exist(Fname, 'file') == 2)
%     str = 'Fname';
%     fprintf('newton(ERROR):no file with name %s exists on path.\n',str);
%     status = -1;
%     return   
% end
if length(x0) <= 0
    str = 'x0';
    fprintf('newton(ERROR):Invalid value for argument %s.\n',str);
    status = -1; 
    return
end
if maxiter < 0
    str = 'maxiter';
    fprintf('newton(ERROR):Invalid value for argument %s.\n',str);
    status = -1; 
    return
end
if printlevel < 0
    str = 'printlevel';
    fprintf('newton(ERROR):Invalid value for argument %s.\n',str);
    status = -1; 
    return
end
if tol <= 0
    str = 'tol';
    fprintf('newton(ERROR):Invalid value for argument %s.\n',str);
    status = -1; 
    return
end

% Constant
TINY = eps^(2/3); % Determines if step is too small to make progress.

% Initialization.
x     = x0;
normx = norm(x);
% [F,J] = feval(Fname,x);
F = funobj.grad(x);
J = funobj.hess(x);
normF = norm(F);

% Print column header and value of F at initial point.
if printlevel ~= 0
  fprintf(' --------------------------------------------------\n');
  fprintf('                   Newton Method                   \n');
  fprintf(' --------------------------------------------------\n');
  fprintf('  Iter      Norm-F         Norm-x       Norm-step  \n')
  fprintf(' %5g %14.7e %14.7e', iter, normF, normx );
end

% Main loop: perform Newton iterations.
while ( 1 )

  % Check for termination
  if (normF <= tol )
     status = 0;
     outcome = ' Stopping tolerance reached';
     break
  elseif (iter >= maxiter )
     status = 2;
     outcome = ' Maximum allowed iterations reached';
     break
  end
      
  % Compute Newton step p.
  p     = - J\F;    
  normp = norm(p);
  
  % Save norm of current iterate before moving updating.
  normxprev = normx;
  
  % Update the iterate and its associated values.
  iter  = iter + 1;
  x     = x + p;
  normx = norm(x);
%   [F,J] = feval(Fname,x);
  F = funobj.grad(x);
  J = funobj.hess(x);
  normF = norm(F);
  
  % Print iterate information, if needed.
  if printlevel ~= 0
    fprintf(' %14.7e \n', normp);                      % End previous line. 
    fprintf(' %5g %14.7e %14.7e', iter, normF, normx); % Start next line.
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
  fprintf( '\n\n Result     :%s \n', outcome)
  if status == 1
    fprintf( '              ||x||    : %13.7e\n', normx)
    fprintf( '              ||step|| : %13.7e\n', normp)
  end
  fprintf( ' Iterations : %-5g\n', iter)
  fprintf( ' Final |F|  : %13.7e\n', normF );
  fprintf(' --------------------------------------------------\n');
end
  
return
