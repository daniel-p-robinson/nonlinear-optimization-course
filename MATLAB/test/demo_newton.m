addpath('../functions/')
addpath('../algorithms/')
fprintf(' Test Rosenbrock...\n')
funobj = Rosenbrock;
x0 = [10;20]; maxiter = 30; printlevel = 1; tol=1e-8;
[x,F,J,iter,status] = newton(funobj, x0, maxiter, printlevel, tol);

% construct the object function by datasetname
fprintf(' Test LeastSquares...\n')
ls = LeastSquares('bodyfat');
x0 = ones(size(ls.A,2), 1);
[x,F,J,iter,status] = newton(ls, x0, maxiter, printlevel, tol);

% construct the object function by datadir + datasetname
fprintf(' Test LeastSquares...\n')
ls = LeastSquares('../data/leastsquares/','bodyfat');
x0 = ones(size(ls.A,2), 1);
[x,F,J,iter,status] = newton(ls, x0, maxiter, printlevel, tol);

% construct the object function by mannually add data
fprintf(' Test LeastSquares...\n')
load('../data/leastsquares/bodyfat.mat', 'A', 'b')
ls = LeastSquares(A, b);
x0 = ones(size(ls.A,2), 1);
[x,F,J,iter,status] = newton(ls, x0, maxiter, printlevel, tol);


fprintf(' Test Logistic...\n')
logit = Logistic('diabetes');
x0 = ones(size(logit.A,2), 1);
[x,F,J,iter,status] = newton(logit, x0, 4, printlevel, tol);

