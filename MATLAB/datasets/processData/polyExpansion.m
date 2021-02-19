% Use the code from https://blog.nus.edu.sg/mattohkc/softwares/SuiteLasso/

function newA = polyExpansion(A, d)
% A: A m-by-n matrix
% d: the degree of the polynomial used in expansion
[m,n] = size(A);
fprintf('Original size: m = %3.0d, n = %3.0d \n', m,n);
v = mypartition(n+1,d);
v = v';
nn = size(v,2);
fprintf('After expansion, n ---> %3.0d\n', nn);
AA = zeros(m,nn);
for q=1:nn
 AAq = ones(m,1);
 for j = 1:n
    if v(j,q) > 0
      AAq = AAq.*(A(:,j).^v(j,q));
    end
 end
 AA(:,q) = AAq;
end
newA = AA;
% remove zero columns
aa = full(sqrt(sum(A.*A)));
idx = find(aa > 0);
if length(idx) < n
  newA = newA(:,idx);
  fprintf('Removed %d zero columns\n', n - length(idx));
end
end

function v = mypartition(n, L1)
% Chose (n-1) the splitting points of the array [0:(n+L1)]
s = nchoosek(1:n+L1-1,n-1);
m = size(s,1);

s1 = zeros(m,1,class(L1));
s2 = (n+L1)+s1;


v = diff([s1 s s2],1,2); 
v = v-1;

end 
