function hstar = find_h_het_PCSE_EC(pra,hL,hU)
% This function calculates the constant h for heteroscedastic errors for
% the case study of Esophageal Cancer
% Using numerical integration with parallel computation

% X is random vector, column vector
% in this Esophageal Cancer Example, X=[X1;X2;X3;X4]
% X1 -age, X2 -risk, X3 -Aspirin reduction factor, X4 -Statin reduction factor
% X1 [55,80] integers, pmf: dispmf(x)=1/26
% X2 [0,0.1] Unif(0,0.1) pdf: f(x) = 10 
% X3 [0,1] Unif(0,1) pdf: f(x) = 1
% X4 [0,1] Unif(0,1) pdf: f(x) = 1

alpha = pra.alpha;

% Waypoints is used to avoid the numerical error in integral when degree of 
% degree of chi square distribution is too high!
% het, df = n0-1
n0 = pra.n0; % initial sample size
chix1 = fzero(@(chix) chi2pdf(chix,n0-1) - chi2pdf(n0-1,n0-1)*(1e-7) ,[0 n0-1]);
chix2 = fzero(@(chix) chi2pdf(chix,n0-1) - chi2pdf(n0-1,n0-1)*(1e-7) ,[n0-1 (n0-1)*3]);
Waypoints_het = [chix1 chix2];
pra.Waypoints_het = round(Waypoints_het);

fh = @(h) f_h(h,pra);
options = optimset('TolX',0.001);
hstar = fzero(@(h) fh(h) - (1-alpha+0.001),[hL hU],options);
end


function f3 = f_h(h,pra)
% X1 - discrete dist, X2,X3,X4 - continuous dist
% X1g = 55:80;
X2g = 0:0.01:0.1;
X3g = 0:0.1:1;
X4g = 0:0.1:1;
[X2m,X3m,X4m] = ndgrid(X2g,X3g,X4g);
f3 = zeros(1,26);
parpool(7); % 7 workers
parfor i = 1:26
    X1dis = i + 54;
    f1 = @(X2,X3,X4) fun_outer(X1dis,X2,X3,X4,h,pra) * 1/26 * 10 * 1 * 1;
    f2 = @(X2,X3,X4) arrayfun(f1,X2,X3,X4);
    F = f2(X2m,X3m,X4m);
    f3(i) = trapz(X4g,trapz(X3g,trapz(X2g,F,1),2),3);
end
delete(gcp)
f3 = sum(f3);
end


%%% Given h, X, compute the outer function
function f_out = fun_outer(X1,X2,X3,X4,h,pra)

m = pra.m; % design points
n0 = pra.n0; % initial sample size
Waypoints_het = pra.Waypoints_het;

X = [X1; X2; X3; X4];
f_out = integral(@(y) f_inner(y,X,h,pra) .* (m * chi2pdf(y,n0-1) .* (1-chi2cdf(y,n0-1)).^(m-1)), 0,Inf, 'Waypoints',Waypoints_het, 'AbsTol',1e-1,'RelTol', 1e-1);

end


%%% Given h, X, y, compute the inner function
function f_in = f_inner(yv,X,h,pra)

L = length(yv);
f_in = zeros(1,L);

k = pra.k; % system number
m = pra.m; % design points
Z = pra.Z;
n0 = pra.n0; % initial sample size
Waypoints_het = pra.Waypoints_het;

for l = 1:L
    y = yv(l);
    fun1 = @(x) ...
        normcdf(1./sqrt(1/y + 1./x) * h/sqrt((n0-1)*([1 X']/(Z'*Z)*[1; X])))...
        .* (m * chi2pdf(x,n0-1) .* (1-chi2cdf(x,n0-1)).^(m-1));
    f_in(l) = integral(fun1,0,Inf, 'Waypoints',Waypoints_het, 'AbsTol',1e-1,'RelTol', 1e-1);
    f_in(l) = f_in(l)^(k-1);
end
end

