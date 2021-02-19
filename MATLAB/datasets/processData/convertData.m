% convert data from libsvm format into matlab .mat format

%% Least Squares
[b, A] = libsvmread('bodyfat_scale.txt');
A = full(A);
save('../LeastSquares/bodyfat.mat', 'A', 'b');

[b, A] = libsvmread('abalone_scale.txt');
save('../LeastSquares/abalone.mat', 'A', 'b');

% polynomial expansion of the original dataset
[b, A] = libsvmread('bodyfat_scale.txt');
A = polyExpansion(A, 3);
save('../LeastSquares/bodyfatExpand3.mat', 'A', 'b');

%% Logistic
[b, A] = libsvmread('diabetes.txt');
A = full(A);
save('../Logistic/diabetes.mat', 'A', 'b');

[b, A] = libsvmread('leu');
save('../Logistic/leu.mat', 'A', 'b');

[b, A] = libsvmread('phishing.txt');
save('../Logistic/phishing.mat', 'A', 'b');