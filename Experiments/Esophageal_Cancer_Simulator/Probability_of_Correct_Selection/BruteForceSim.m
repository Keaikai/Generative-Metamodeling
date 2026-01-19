clear, clc;

repN = 500000;
CV = 1;

s_pseudo = RandStream('mt19937ar','Seed',0);
RandStream.setGlobalStream(s_pseudo); 

parpool(7); % 7 workers

% no drug
riskR = 0:0.01:0.1;
ztemp = zeros(length(riskR),2);
QALY_nodrug = zeros(length(riskR),2,26);
for age = 55 : 80
    parfor i = 1:length(riskR)
        fprintf('no drug, age = %d,  i = %d:  ',age,i);
        risk = riskR(i);
        tic
        ztemp(i,:) = EsophagealCancerSim(risk,[],[],0,age,repN,CV,'mean_var');
        toc
        fprintf('\n');
    end
    QALY_nodrug(:,:,age-54) = ztemp;
end


% use aspirin
AspirinR = 0:0.1:1;
[X1,X2] = ndgrid(riskR,AspirinR);
X = [X1(:) X2(:)];
ztemp = zeros(size(X,1),2);
QALY_aspirin = zeros(size(X,1),2,26);
for age = 55 : 80
    parfor i = 1:size(X,1)
        fprintf('drug1-aspirin, age = %d,  i = %d:  ',age,i);
        risk = X(i,1);
        aspirin = X(i,2);
        tic
        ztemp(i,:) = EsophagealCancerSim(risk,aspirin,0,1,age,repN,CV,'mean_var');
        toc
        fprintf('\n');
    end
    QALY_aspirin(:,:,age-54) = ztemp;
end


% use statin
StatinR = 0:0.1:1;
[X1,X2] = ndgrid(riskR,StatinR);
X = [X1(:) X2(:)];
ztemp = zeros(size(X,1),2);
QALY_statin = zeros(size(X,1),2,26);
for age = 55 : 80
    parfor i = 1:size(X,1)
        fprintf('drug2-statin, age = %d,  i = %d:  ',age,i);
        risk = X(i,1);
        statin = X(i,2);
        tic
        ztemp(i,:) = EsophagealCancerSim(risk,0,statin,2,age,repN,CV,'mean_var');
        toc
        fprintf('\n');
    end
    QALY_statin(:,:,age-54) = ztemp;
end

delete(gcp)    

save('QALY_true.mat','QALY_nodrug','QALY_aspirin','QALY_statin');


%%% Check the half-width of the 95% confidence interval
% 100(1-alpha)% CI

% alpha = 0.05;
% z = norminv(1-alpha/2,0,1) % 1.96
% t = tinv(1-alpha/2,repN-1) % 1.96

a = max(max(QALY_nodrug(:,2,:)));
b = max(max(QALY_aspirin(:,2,:)));
c = max(max(QALY_statin(:,2,:)));
max_half_width = 1.96 * sqrt(max([a,b,c])) / sqrt(repN)