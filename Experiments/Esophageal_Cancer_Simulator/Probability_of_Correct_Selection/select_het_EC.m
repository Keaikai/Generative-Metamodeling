function [thetahat, totalN] = select_het_EC(h,pra)

% system number
k = pra.k;
% dimension
d = pra.d;
% design points
m = pra.m;
Xd = pra.Xd;
Z = pra.Z;
% initial sample size
n0 = pra.n0;
% indifferent-zone parameter
delta = pra.delta;


%%%%%% Step 0 - solve h

%%%%%% Step 1
% run n0 batches for k systems
Y = kSystem_EC(Xd,0,n0); % n0 batches

Ybar = mean(Y,3);
S2mk = zeros(m,k);
for i = 1:k
    for q = 1:m
        S2mk(q,i) = sum((Y(q,i,:) - Ybar(q,i)).^2) / (n0-1);
    end
end
N = h^2 * S2mk / (delta^2);
N = ceil(N);
N = max(N, n0*ones(m,k));

%%%%%% Step 2
% run Nqi-n0 observations of Yi on design point xm
for i = 1:k
    for q = 1:m % m design points
        if N(q,i) > n0
            Y(q,i,n0+1:N(q,i))= kSystem_EC(Xd(q,:),i,N(q,i)-n0);       
        end
    end
end
Yhat = zeros(m,k);

for i = 1:k
    for q = 1:m
        Yhat(q,i) = mean(Y(q,i,1:N(q,i)));
    end
end

thetahat = zeros(d+1,k);
for i=1:k
    thetahat(:,i) = (Z'*Z)\Z'*Yhat(:,i);  
end

totalN = sum(sum(N));

end