[b, A] = libsvmread('bodyfat_scale.txt');
A = full(A);
save('../data/bodyfat.mat', 'A', 'b');

[b, A] = libsvmread('diabetes.txt');
A = full(A);
save('../data/diabetes.mat', 'A', 'b');