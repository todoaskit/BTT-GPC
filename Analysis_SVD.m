%T = xlsread('totaldataset.xlsx');
Tref = xlsread('refinedData.xlsx');
[n, m] = size(Tref);
Tnorm = zeros(n,m); 
%Tnorm = (Tref - min(min(Tref)))/(max(max(Tref)) - min(min(Tref)));

% 
for i=1:n
    tmpmax = max(Tref(i, :));
    tmpmin = min(Tref(i, :));
    if tmpmax > tmpmin
        Tnorm(i, :) = (Tref(i, :) - tmpmin)/(tmpmax-tmpmin);
    end
end
%[U, S, V] = svd(T, 'econ');
[Ur, Sr, Vr] = svd(Tref, 'econ');
[Un, Sn, Vn] = svd(Tnorm, 'econ');
plot(log10(diag(Sn))); title('Singular values of the normalized data matrix', 'Fontsize', 15)
% plot(log10(diag(Sr))); title('Singular values of the data matrix', 'Fontsize', 15)
ylabel('log_{10}\sigma_i', 'Fontsize', 15)


%% low rank approximation of raw data T with rank r < m
r = 80;
S_reduced = zeros(m,m);
S_reduced(1:r,1:r) = Sn(1:r,1:r);
T_reduced = Un*S_reduced*Vn';
[Unn, Snn, Vnn] = svd(T_reduced);