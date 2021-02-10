addpath('../functions/')
%% LeastSquare
%%% create data mannually
fprintf('Testing LeastSquares...\n')
clear
fprintf(' Mannual Input Data...\n')
load('../data/LeastSquares/bodyfat.mat')
ls = LeastSquares(A, b);
x0 = ones(size(A,2), 1);
f = ls.func(x0);
g = ls.grad(x0);
h = ls.hess(x0);
v0 = 2 * x0;
hv = ls.hvprod(x0, v0);
if norm(hv - h * v0, 'inf') < 1e-10
    fprintf('  hv: matched!\n')
else
    fprintf('  hv: wrong!\n')
end
%%% load data from files
clear
fprintf(' load data from mat files...\n')
ls = LeastSquares('bodyfat');
x0 = ones(size(ls.A,2), 1);
f = ls.func(x0);
g = ls.grad(x0);
h = ls.hess(x0);
v0 = 2 * x0;
hv = ls.hvprod(x0, v0);
if norm(hv - h * v0, 'inf') < 1e-10
    fprintf('  hv: matched!\n')
else
    fprintf('  hv: wrong!\n')
end

%% Logistic
%%% create data mannually
fprintf('Testing Logistic...\n')
clear
fprintf(' Specify Input Data...\n')
load('../data/logistic/diabetes.mat')
logit = Logistic(A, b);
x0 = ones(size(A,2), 1);
f = logit.func(x0);
g = logit.grad(x0);
h = logit.hess(x0);
v0 = 2 * x0;
hv = logit.hvprod(x0, v0);
if norm(hv - h * v0, 'inf') < 1e-10
    fprintf('  hv: matched!\n')
else
    fprintf('  hv: wrong!\n')
end
%%% load data from files
clear
fprintf(' load data from mat files...\n')
logit = Logistic('diabetes');
x0 = ones(size(logit.A,2), 1);
f = logit.func(x0);
g = logit.grad(x0);
h = logit.hess(x0);
v0 = 2 * x0;
hv = logit.hvprod(x0, v0);
if norm(hv - h * v0, 'inf') < 1e-10
    fprintf('  hv: matched!\n')
else
    fprintf('  hv: wrong!\n')
end

try
    logit = Logistic('bodyfat');
catch E
    fprintf(1,'There was an error! The message was:\n%s\n',E.message);
end

