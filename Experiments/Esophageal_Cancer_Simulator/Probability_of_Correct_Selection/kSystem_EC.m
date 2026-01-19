function Y = kSystem_EC(X,runi,repN)

% k=2, using aspirin or statin
% X is the design matrix, X=[x1; x2; ... xm]
% x1, x2, .... xm, are m design points; each one is row vector
% x = [iniage,risk,reduction_Aspirin,reduction_Statin]
% runi == 0, run all systems; otherwise, only run system i
% repN, how many repeated observations

m = size(X,1); % number of design points

if runi == 0 % run all systems
    Y = zeros(m,3,repN);
    for sysi = 1:2    % drug = sysi-1
        for j = 1:m   % design point j
            Y(j,sysi,:) = EsophagealCancerSim(X(j,2),X(j,3),X(j,4),sysi,X(j,1),repN,0,'raw');
        end
    end

else % run system i
    Y = zeros(m,1,repN);
    for j = 1:m  % design point j
        Y(j,1,:) = EsophagealCancerSim(X(j,2),X(j,3),X(j,4),runi,X(j,1),repN,0,'raw');
    end
end

end