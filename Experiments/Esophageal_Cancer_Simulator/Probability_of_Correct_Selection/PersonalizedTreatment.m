clear,clc; close all

% =========================
%  Create ALL folders once (before any computation)
%  Put every directory that will be used later (csvwrite / save / load) here.
%  =========================

% base folders
dataDir = './data';
rsDir   = fullfile(dataDir, 'RS');

% ---- list of directories to ensure (column cell, safest) ----
dirList = {
    fullfile(rsDir,'Xsample')        % csvwrite('./data/RS/Xsample/Xsample.csv')
    fullfile(rsDir,'results')        % PCS_*.csv
    fullfile(rsDir,'ECtraindata1')   % ECtraindata1_*.csv
    fullfile(rsDir,'ECtraindata2')
    fullfile(rsDir,'traindata1')     % traindata1_*.csv
    fullfile(rsDir,'traindata2')
    fullfile(rsDir,'RS_CWGAN')       % choice_*.csv
    fullfile(rsDir,'RS_Diffusion')
    fullfile(rsDir,'RS_RectFlow')
    fullfile(rsDir,'QRGMMcoeff1')
    fullfile(rsDir,'QRGMMcoeff2')
};

% ---- create if not exists ----
for k = 1:numel(dirList)
    if ~exist(dirList{k}, 'dir')
        mkdir(dirList{k});
    end
end

fprintf('[Folder setup] All required directories are ensured.\n');


%-------- Begin Experiments--------% 

k = 2; % system number, 1 - Aspirin, 2 - Statin
d = 4; % dimension, x = [age,risk,reduction_Aspirin,reduction_Statin]
repn = 100; % the replication number 

n0 = 100; % initial sample size
delta = 1/6; % indifferent-zone parameter (2 months)
alpha = 0.05;  % PCS = 1-alpha = 95%



%-------- TS Procedure --------% 


m = 16; % design points (full factorial design)
Xd = [61 0.1/4 1/4 1/4; ...  % every point, row vector
      61 0.1/4 1/4 3/4; ...
      61 0.1/4 3/4 1/4; ...
      61 0.1/4 3/4 3/4; ...
      61 0.3/4 1/4 1/4; ...
      61 0.3/4 1/4 3/4; ...
      61 0.3/4 3/4 1/4; ...
      61 0.3/4 3/4 3/4; ...
      74 0.1/4 1/4 1/4; ...
      74 0.1/4 1/4 3/4; ...
      74 0.1/4 3/4 1/4; ...
      74 0.1/4 3/4 3/4; ...
      74 0.3/4 1/4 1/4; ...
      74 0.3/4 1/4 3/4; ...
      74 0.3/4 3/4 1/4; ...
      74 0.3/4 3/4 3/4];    
Z = [ones(m,1) Xd];

pra.k = k; pra.d = d; pra.m = m; pra.Xd = Xd; pra.Z = Z;
pra.n0 = n0; pra.delta = delta; pra.alpha = alpha;

s_pseudo = RandStream('mt19937ar','Seed',100);
RandStream.setGlobalStream(s_pseudo); 
%% solve h - EPCS
tic
h_het = find_h_het_PCSE_EC(pra,1,2)
toc

%h_het = 1.6972; %by solving find_h_het_PCSE_EC(pra,1,2), using time 2,8372.044886 seconds(7 workers)

thetahat_het = zeros(1+d,k,repn);
N_het = zeros(repn,1);
for rep = 1:repn
    tic
    [thetahat_het(:,:,rep), N_het(rep)] = select_het_EC(h_het,pra);
end
save('data_het.mat','thetahat_het','N_het');
fprintf('het is done \n');

%load('data_het.mat')


%%%%%  Evaluation  %%%%%%

% "true" value grid
ageR = 55:80;
riskR = 0:0.01:0.1;
AspirinR = 0:0.1:1;
StatinR = 0:0.1:1;
load('QALY_true.mat') % from Brute Force Simulation

% randomly take covariates according to its distribution
s_pseudo = RandStream('mt19937ar','Seed',100);
RandStream.setGlobalStream(s_pseudo);
Xn = 100;
Xsample = zeros(Xn, 4);

Xsample(:,1)=randi([55 80],Xn,1);
Xsample(:,2) =0.1*rand(Xn,1);
Xsample(:,3)=rand(Xn,1);
Xsample(:,4)=rand(Xn,1);

filename=strcat('./data/RS/Xsample/Xsample.csv');
csvwrite(filename,Xsample);


%%%% the "true" best %%%%
% linear interpolation
[G1,G2] = ndgrid(riskR,AspirinR);
GG = [G1(:) G2(:)]; % the same for Statin
y_true = zeros(Xn,2);
for i = 1:Xn  
    X = Xsample(i,:);
    age = X(1);
    
    x1u = ceil(X(2)*100)/100; x1l = floor(X(2)*100)/100;
    x2u = ceil(X(3)*10)/10; x2l = floor(X(3)*10)/10;
    x3u = ceil(X(4)*10)/10; x3l = floor(X(4)*10)/10;    
    if x1u == x1l % risk is integer
        s1 = 0.5;
    else
        s1 = (X(2)-x1u) / (x1l-x1u);
    end
    if x2u == x2l % aspirin is integer
        s2 = 0.5;
    else
        s2 = (X(3)-x2u) / (x2l-x2u);
    end    
    if x3u == x3l % statin is integer
        s3 = 0.5;
    else
        s3 = (X(4)-x3u) / (x3l-x3u);
    end 
    
    
    % system 1, Aspirin
    row = find( sum(abs(GG-repmat([x1l x2l],size(GG,1),1)),2) < 0.001 );
    y_x1lx2l = QALY_aspirin(row,1,age-54);    
    row = find( sum(abs(GG-repmat([x1u x2l],size(GG,1),1)),2) < 0.001 );
    y_x1ux2l = QALY_aspirin(row,1,age-54);    
    row = find( sum(abs(GG-repmat([x1l x2u],size(GG,1),1)),2) < 0.001 );
    y_x1lx2u = QALY_aspirin(row,1,age-54);       
    row = find( sum(abs(GG-repmat([x1u x2u],size(GG,1),1)),2) < 0.001 );
    y_x1ux2u = QALY_aspirin(row,1,age-54);        
    y_x2l = y_x1lx2l*s1 + y_x1ux2l*(1-s1);
    y_x2u = y_x1lx2u*s1 + y_x1ux2u*(1-s1);    
    y_true(i,1) = y_x2l*s2 + y_x2u*(1-s2);
    
    % system 2, Statin 
    row = find( sum(abs(GG-repmat([x1l x3l],size(GG,1),1)),2) < 0.001 );
    y_x1lx3l = QALY_statin(row,1,age-54);    
    row = find( sum(abs(GG-repmat([x1u x3l],size(GG,1),1)),2) < 0.001 );
    y_x1ux3l = QALY_statin(row,1,age-54);    
    row = find( sum(abs(GG-repmat([x1l x3u],size(GG,1),1)),2) < 0.001 );
    y_x1lx3u = QALY_statin(row,1,age-54);       
    row = find( sum(abs(GG-repmat([x1u x3u],size(GG,1),1)),2) < 0.001 );
    y_x1ux3u = QALY_statin(row,1,age-54);           
    y_x3l = y_x1lx3l*s1 + y_x1ux3l*(1-s1);
    y_x3u = y_x1lx3u*s1 + y_x1ux3u*(1-s1);    
    y_true(i,2) = y_x3l*s3 + y_x3u*(1-s3);        
end
[max_true, choice_true] = max(y_true,[],2);


%-------- TS Procedure --------% 
CorrR_het_E = zeros(1,repn);
tic
for rep = 1:repn
    %%% evaluate the AEPCS
    yestimate = [ones(Xn,1) Xsample] * thetahat_het(:,:,rep); % for entire test points
    [~, choice] = max(yestimate,[],2);
    index = sub2ind(size(y_true),(1:Xn)',choice);
    correct = choice == choice_true | abs(y_true(index) - max_true) < delta - 1e-6;
    CorrR_het_E(rep) = sum(correct) / Xn;  
end
APCSE_TS = mean(CorrR_het_E)
toc % 0.016419 秒
filename=strcat('./data/RS/results/PCS_TS.csv');
csvwrite(filename,APCSE_TS);


%-------- KN Procedure + Simulator --------% 
CorrR_het_E_KN_Sim = zeros(1,repn);
Sample_size_KN_Sim = zeros(1,repn);
tic
for rep = 1:repn
    [choice,sample_size] = KN_simulator(Xsample,k,alpha,delta,n0);
    index = sub2ind(size(y_true),(1:Xn)',choice);
    correct = choice == choice_true | abs(y_true(index) - max_true) < delta - 1e-6;
    CorrR_het_E_KN_Sim(rep) = sum(correct) / Xn;
    Sample_size_KN_Sim(rep) = sum(sample_size)/ Xn;

end
toc
APCSE_KN_Sim = mean(CorrR_het_E_KN_Sim)
mean_Sample_size_KN_Sim = mean(Sample_size_KN_Sim)
filename=strcat('./data/RS/results/PCS_KN_Sim.csv');
csvwrite(filename,APCSE_KN_Sim);
%历时 19399.913242 秒。mean_Sample_size_KN_Sim =1.1718e+04

%-------- Generate data for GMM --------% 

for runi = 1:repn
    rng(runi)
    nsample=100000;
    X1=randi([55 80],nsample,1);
    X2=0.1*rand(nsample,1);
    X3=rand(nsample,1);
    X4=rand(nsample,1);

    X=[X1 X2 X3 X4];
    PX=Poly(X,2);

    GMMtraindata1=zeros(size(PX,1),size(PX,2)+2);
    GMMtraindata1(:,2:size(PX,2)+1)=PX;
    GMMtraindata1(:,1)=1;

    GMMtraindata2=zeros(size(PX,1),size(PX,2)+2);
    GMMtraindata2(:,2:size(PX,2)+1)=PX;
    GMMtraindata2(:,1)=1;


    for i=1:size(PX,1)
        GMMtraindata1(i,end) = EsophagealCancerSim(X(i,2),X(i,3),X(i,4),1,X(i,1),1,0,'raw');
    end

    for i=1:size(PX,1)
        GMMtraindata2(i,end) = EsophagealCancerSim(X(i,2),X(i,3),X(i,4),2,X(i,1),1,0,'raw');
    end

    filename1=strcat('./data/RS/ECtraindata1/ECtraindata1_',num2str(runi),'.csv');
    csvwrite(filename1,GMMtraindata1);

    filename2=strcat('./data/RS/ECtraindata2/ECtraindata2_',num2str(runi),'.csv');
    csvwrite(filename2,GMMtraindata2);
 
    traindata1=[GMMtraindata1(:,1:5),GMMtraindata1(:,end)];
    filename1=strcat('./data/RS/traindata1/traindata1_',num2str(runi),'.csv');
    csvwrite(filename1,traindata1);  

    traindata2=[GMMtraindata2(:,1:5),GMMtraindata2(:,end)];
    filename2=strcat('./data/RS/traindata2/traindata2_',num2str(runi),'.csv');
    csvwrite(filename2,traindata2); 
end

%-------- QRGMM for R&S --------% 

global le;
global ue;
global quantilepoints
global coefficients1
global coefficients2
m=320;
le=1/m;
ue=1-le;
quantilenum=m;
quantilepoints=linspace(le,ue,m-1);

CorrR_het_E_RS_QRGMM = zeros(1,repn);
s_pseudo = RandStream('mt19937ar','Seed',100);
RandStream.setGlobalStream(s_pseudo);

for rep = 1:repn
    filename=strcat('./data/RS/GMMcoeff1/GMMcoeff1_',num2str(rep),'.csv');
    coefficients1=csvread(filename,1,1);
    filename=strcat('./data/RS/GMMcoeff2/GMMcoeff2_',num2str(rep),'.csv');
    coefficients2=csvread(filename,1,1);
    [choice] = RS_QRGMM(Xsample);
    index = sub2ind(size(y_true),(1:Xn)',choice);
    correct = choice == choice_true | abs(y_true(index) - max_true) < delta - 1e-6;
    CorrR_het_E_RS_QRGMM(rep) = sum(correct) / Xn;
end

APCSE_RS_QRGMM = mean(CorrR_het_E_RS_QRGMM)
filename=strcat('./data/RS/results/PCS_RS_QRGMM.csv');
csvwrite(filename,APCSE_RS_QRGMM);



%-------- CWGAN for R&S --------% 
CorrR_het_E_RS_CWGAN = zeros(1,repn);
for rep = 1:repn
    %%% evaluate the AEPCS
    filename=strcat('./data/RS/RS_CWGAN/choice_',num2str(rep),'.csv');
    choice=csvread(filename,0,0);
    index = sub2ind(size(y_true),(1:Xn)',choice);
    correct = choice == choice_true | abs(y_true(index) - max_true) < delta - 1e-6;
    CorrR_het_E_RS_CWGAN(rep) = sum(correct) / Xn;  
end
APCSE_RS_CWGAN = mean(CorrR_het_E_RS_CWGAN)
filename=strcat('./data/RS/results/PCS_RS_CWGAN.csv');
csvwrite(filename,APCSE_RS_CWGAN);


%-------- Diffusion for R&S --------% 
CorrR_het_E_RS_Diffusion = zeros(1,repn);
for rep = 1:repn
    %%% evaluate the AEPCS
    filename=strcat('./data/RS/RS_Diffusion/choice_',num2str(rep),'.csv');
    choice=csvread(filename,0,0);
    index = sub2ind(size(y_true),(1:Xn)',choice);
    correct = choice == choice_true | abs(y_true(index) - max_true) < delta - 1e-6;
    CorrR_het_E_RS_Diffusion(rep) = sum(correct) / Xn;  
end
APCSE_RS_Diffusion = mean(CorrR_het_E_RS_Diffusion)
filename=strcat('./data/RS/results/PCS_RS_Diffusion.csv');
csvwrite(filename,APCSE_RS_Diffusion);


%-------- RectFlow for R&S --------% 
CorrR_het_E_RS_RectFlow = zeros(1,repn);
for rep = 1:repn
    %%% evaluate the AEPCS
    filename=strcat('./data/RS/RS_RectFlow/choice_',num2str(rep),'.csv');
    choice=csvread(filename,0,0);
    index = sub2ind(size(y_true),(1:Xn)',choice);
    correct = choice == choice_true | abs(y_true(index) - max_true) < delta - 1e-6;
    CorrR_het_E_RS_RectFlow(rep) = sum(correct) / Xn;  
end
APCSE_RS_RectFlow = mean(CorrR_het_E_RS_RectFlow)
filename=strcat('./data/RS/results/PCS_RS_RectFlow.csv');
csvwrite(filename,APCSE_RS_RectFlow);

