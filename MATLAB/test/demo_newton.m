addpath('../functions/')
addpath('../algorithms/')
fprintf('\n Testing Rosenbrock...\n')
funobj = Rosenbrock;
x0 = [10;20]; maxiter = 30; printlevel = 1; tol=1e-10;
[x,F,J,iter,status] = newton(funobj, x0, maxiter, printlevel, tol);

% construct the object function by datasetname
fprintf('\n  Testing LeastSquares...\n')
ls = LeastSquares('bodyfat');
x0 = ones(size(ls.A,2), 1);
[x,F,J,iter,status] = newton(ls, x0, maxiter, printlevel, tol);

% construct the object function by datadir + datasetname
fprintf('\n Testing LeastSquares...\n')
ls = LeastSquares('../data/leastsquares/','bodyfat');
x0 = ones(size(ls.A,2), 1);
[x,F,J,iter,status] = newton(ls, x0, maxiter, printlevel, tol);

% construct the object function by mannually add data
fprintf('\n  Testing LeastSquares...\n')
load('../data/leastsquares/bodyfat.mat', 'A', 'b')
ls = LeastSquares(A, b);
x0 = ones(size(ls.A,2), 1);
[x,F,J,iter,status] = newton(ls, x0, maxiter, printlevel, tol);


fprintf('\n Testing Logistic...\n')
logit = Logistic('diabetes');
x0 = ones(size(logit.A,2), 1);
[x,F,J,iter,status] = newton(logit, x0, 4, printlevel, tol);

