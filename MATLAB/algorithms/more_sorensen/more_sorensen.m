function [x,info] = more_sorensen(H,g,radius,params)
% ========================================================================
% This function implements the trust region subproblem solver of More and
% Sorensen, which either yields the unconstrained minimizer of the
% quadratic objective or applies Newton's method to find the multiplier of
% the trust region constraint that yields the desired norm of the step.
% A valid call for using this function is of the form
%      [x,info] = more_sorensen(H,g,radius,params)
% where H is a square and symmetric matrix of size n-by-n, g is vector of
% length n, radius is a positive trust region radius, and params is a
% structure that may contain input parameters (see below).
% ========================================================================
% Author      : Frank E. Curtis and Daniel P. Robinson
% Description : More-Sorensen trust region subproblem solver
% Input       : H       ~ matrix defining quadratic term
%               b       ~ vector defining linear term
%               radius  ~ radius defining trust region
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
%                      lambda  multiplier for trust region constraint
%                      r_norm  norm of residual of optimality conditions
% ========================================================================

% Initialize output
x = [];
info.status = -1;
info.outcome = 'Insufficient number of inputs.';
info.k = 0;
info.lambda = [];
info.r_norm = [];

% Make sure enough input parameters are supplied.
if nargin < 4
  fprintf('\n more_sorensen(ERROR):not enough input parameters.\n');
  return
end

% Set problem size
[n,m] = size(g);

% Check sizes of inputs
if m ~= 1 || size(H,1) ~= n || size(H,2) ~= n || size(radius,1) ~= 1 || size(radius,2) ~= 1
  fprintf('\n more_sorensen(ERROR):invalid input parameters.\n');
  return
end

% Initialize output
x = zeros(length(g),1);
info.status = -1;
info.outcome = 'Invalid input parameter.';
info.k = 0;
info.lambda = 0;
info.r_norm = computeResidual(H,g,radius,x,info.lambda);

% Make sure the final input, which holds the control parameters, is okay.
% If any of the required fields are not supplied, set default values.
if ~isa(params,'struct')
  str = 'params';
  fprintf('\n more_sorensen(ERROR):Invalid input parameter %s.\n',str);
  return
end
str2 = 'params';
if isfield(params,'tol')
  tol = params.tol;
  if any(tol < 0)
    str1 = 'tol';
    fprintf('\n more_sorensen(ERROR):Invalid field %s in input %s.\n',str1,str2);
    return
  end
else
  tol = 1e-6;
end
if isfield(params,'maxiter')
  maxiter = params.maxiter;
  if maxiter < 0
    str1 = 'maxiter';
    fprintf('\n more_sorensen(ERROR):Invalid field %s in input %s.\n',str1,str2);
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
    fprintf('\n more_sorensen(ERROR):Invalid control parameter %s.\n',str);
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
    fprintf('\n more_sorensen(ERROR):Invalid control parameter %s.\n',str);
    info.status = -1;
    return
  end
else
  probname = '';
end

% Store norm of g
g_norm2 = norm(g);

% Compute dual variable bound values (see CGT pg. 192)
G_vec     =  abs(H)*ones(n,1) - abs(diag(H));
G_bnds(1) =  max( diag(H)+G_vec);
G_bnds(2) =  max(-diag(H)+G_vec);
H_bnds    =  min(norm(H,'fro'),norm(H,inf));
l_max     =  min(G_bnds(1),H_bnds);
l_min     = -min(G_bnds(2),H_bnds);
l_init    =  g_norm2/radius;

% Compute minimum eigenpair
[eig_val,eig_vec] = computeMinimumEigenpair(n,H,0);

% Replace minimum eigenvalue estimate
l_min = max(l_min,eig_val);

% Set dual variable bounds (see CGT pg. 192)
lambdaL = full(max([0,l_init - l_max,-min(diag(H)),-eig_val]));
lambdaU = max([0,l_init - l_min]);

% Set estimate
info.lambda = lambdaL;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STEPS INDICATED IN ALGORITHM 7.3.4 (pg. 193) FROM CGT BOOK %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize factorization boolean
fact = 0;

% Store output strings
out_line = '=============================================';
out_data = '  k        lambda       ||x||        ||r||';

% Print output header
if printlevel >= 1
    fprintf(outfileID,'%s\n',out_line);
    fprintf(outfileID,'More-Sorensen Trust Region Subproblem Solver\n');
    fprintf(outfileID,'%s\n',out_line);
    fprintf(outfileID,' maximum iterations    : %g\n',maxiter);
    fprintf(outfileID,' print level           : %g\n',printlevel);
    fprintf(outfileID,' file for output       : %s\n',outfileNAME);
    fprintf(outfileID,' problem name          : %s\n',probname);
    fprintf(outfileID,'%s\n%s\n%s\n',out_line,out_data,out_line);
end

% Start iteration counter
info.k = 1;

% Iteration loop
while 1
  
  %%%%%%%%%%
  % STEP 1 %
  %%%%%%%%%%
  
  % Check for factorization completed
  if fact == 0
    
    % Factorize shifted Hessian
    [L,err] = chol(H+info.lambda*speye(n),'lower');
    
  end
  
  % Check for error
  if err == 0
    
    % Solve system
    ls_forw = -(L\g);
    x       = (ls_forw'/L)';
    
    % Evaluate step norm
    x_norm = norm(x);
    
  end
  
  % Print iteration information
  if printlevel >= 1
    info.r_norm = computeResidual(H,g,radius,x,info.lambda);
    if err == 0
      fprintf(outfileID,'%6d  %+.4e  %+.4e  %+.4e\n',info.k,info.lambda,x_norm,info.r_norm);
    else
      fprintf(outfileID,'%6d  %+.4e  %+.4e  %+.4e\n',info.k,info.lambda,'----------',info.r_norm);
    end
  end
  
  % Check for interior convergence
  if err == 0 && info.lambda == 0 && x_norm <= radius
    info.status = 0;
    info.outcome = 'Optimal (interior) solution found.';
    info.r_norm = computeResidual(H,g,radius,x,info.lambda);
    break
  end
  
  %%%%%%%%%%
  % STEP 2 %
  %%%%%%%%%%
  
  % Update upper or lower bound
  if err == 0 && x_norm <= radius
    lambdaU = info.lambda;
  else
    lambdaL = info.lambda;
  end
  
  %%%%%%%%%%
  % STEP 3 %
  %%%%%%%%%%
  
  % Check for error
  if err == 0
    
    % Solve system for Newton step
    ls_temp = L\x;
    
    % Set trial dual variable
    lambdaN = info.lambda + (x_norm/norm(ls_temp))^2*((x_norm-radius)/radius);
    
    % Check for short step
    if x_norm <= radius
      
      % Compute minimum eigenpair
      [~,eig_vec] = computeMinimumEigenpair(n,H,info.lambda);
      
      % Compute inner product
      uHu = full(eig_vec'*(H*eig_vec) + info.lambda*norm(eig_vec)^2);
      
      % Update lower bound
      lambdaL = max(lambdaL,info.lambda-uHu);
      
      % Compute roots
      alphas = roots([norm(eig_vec)^2,2*x'*eig_vec,x_norm^2-radius^2]);
      
      % Choose best root
      if g'*(x+alphas(1)*eig_vec) + (1/2)*(x+alphas(1)*eig_vec)'*H*(x+alphas(1)*eig_vec) < ...
          g'*(x+alphas(2)*eig_vec) + (1/2)*(x+alphas(2)*eig_vec)'*H*(x+alphas(2)*eig_vec)
        alpha = alphas(1);
      else
        alpha = alphas(2);
      end
      
    end
    
  else
    
    % Compute minimum eigenpair
    [eig_val,eig_vec] = computeMinimumEigenpair(n,H,info.lambda);
    
    % Update lower bound
    lambdaL = full(max(lambdaL,-eig_val));
    
  end
  
  %%%%%%%%%%
  % STEP 4 %
  %%%%%%%%%%
  
  % Check for termination
  if (err == 0 && abs(radius - x_norm) <= tol*radius) || ...
      (err == 0 && x_norm <= radius && info.lambda == 0)
    info.status = 0;
    info.outcome = 'Optimal (boundary) solution found.';
    info.r_norm = computeResidual(H,g,radius,x,info.lambda);
    break
  end
  if info.k >= maxiter
    info.status = 0;
    info.outcome = 'Iteration limit reached.';
    info.r_norm = computeResidual(H,g,radius,x,info.lambda);
    break
  end
  if (err == 0 && x_norm <= radius && alpha^2*uHu <= tol*(x'*(H*x) + info.lambda*x_norm^2 + info.lambda*radius^2))
    x = x + alpha*eig_vec;
    info.status = 0;
    info.outcome = 'Optimal (boundary) solution found.';
    info.r_norm = computeResidual(H,g,radius,x,info.lambda);
    break
  end
  
  %%%%%%%%%%
  % STEP 5 %
  %%%%%%%%%%
  
  % Check step norm
  if err == 0 && x_norm > radius && g_norm2 > 0
    info.lambda = lambdaN;
    fact = 0;
  elseif err == 0 && x_norm <= radius
    
    % Attempt to factorize new system
    [L,err] = chol(H+lambdaN*speye(n),'lower');
    
    % Check error
    if err == 0
      fact = 1;
      info.lambda = lambdaN;
    else
      
      % Reset boolean
      fact = 0;
      
      % Update lower bound
      lambdaL = max(lambdaL,lambdaN);
      
      % Attempt to factorize system with lambdaL
      [L,err] = chol(H+lambdaL*speye(n),'lower');
      
      % Check for error
      if err == 0
        
        % Solve system
        ls_forw = -(L\g);
        x       = (ls_forw'/L)';
        
        % Evaluate step norm
        x_norm = norm(x);
        
      end
      
      % Check for interior convergence
      if err == 0 && x_norm <= radius
        
        % Update dual variable
        info.lambda = lambdaL;
        
        % Compute minimum eigenpair
        [~,eig_vec] = computeMinimumEigenpair(n,H,info.lambda);
        
        % Compute roots
        alphas = roots([norm(eig_vec)^2,2*x'*eig_vec,x_norm^2-radius^2]);
        
        % Choose best root
        if g'*(x+alphas(1)*eig_vec) + (1/2)*(x+alphas(1)*eig_vec)'*H*(x+alphas(1)*eig_vec) < ...
            g'*(x+alphas(2)*eig_vec) + (1/2)*(x+alphas(2)*eig_vec)'*H*(x+alphas(2)*eig_vec)
          alpha = alphas(1);
        else
          alpha = alphas(2);
        end
        
        % Set new solution
        x = x + alpha*eig_vec;
        
        % Set step norm
        x_norm = norm(x);
        
      else
        
        % Update dual variable
        info.lambda = max(sqrt(lambdaL*lambdaU),lambdaL + 0.5*(lambdaU - lambdaL));
        
      end
      
    end
    
  else
    
    % Reset dual variable estimate
    info.lambda = max(sqrt(lambdaL*lambdaU),lambdaL + 0.5*(lambdaU - lambdaL));
    
  end
  
  % Increment iteration counter
  info.k = info.k + 1;
  
end

% Print output footer
fprintf(outfileID,'%s\n',out_line);
fprintf(outfileID,'Number of iterations  : %g\n', info.k);
fprintf(outfileID,'Status on termination : %-2g\n', info.status);
fprintf(outfileID,'Outcome               : %s\n', info.outcome);
fprintf(outfileID,'Final residual norm   : %13.7e\n', info.r_norm); 
fprintf(outfileID,'%s\n',out_line);

end

% Compute minimum eigenvalue and corresponding eigenvector
function [eig_val,eig_vec] = computeMinimumEigenpair(n,H,lambda)

% Compute using eigs
[eig_vec,eig_val] = eigs(H+lambda*speye(n),1,'SA');

% Recompute using eig if eigs failed
if isnan(eig_val)
  [EIGVEC,EIGVAL] = eig(full(H+lambda*speye(n)));
  eigvals = diag(EIGVAL);
  [eig_val,index] = min(eigvals);
  eig_vec = EIGVEC(:,index);
end

end

% Compute residual of optimality conditions
function r_norm = computeResidual(H,g,radius,x,lambda)

% Compute minimum eigenpair
[eig_val,~] = computeMinimumEigenpair(length(g),H,lambda);

% Compute infinity norm of conditions
r_norm = norm([g + (H + lambda*eye(length(g)))*x;
               min(eig_val,0);
               min(lambda,0);
               min(radius - norm(x),0);
               lambda*(radius - norm(x))],inf);

end