clear
clc

%% ---- Create all output folders once (before the loop) ----
baseDir = './data';

dirList = { ...
    fullfile(baseDir,'QRGMMoutput1_1'), ...
    fullfile(baseDir,'QRGMMoutput1_2'), ...
    fullfile(baseDir,'QRGMMoutputtest1'), ...
    fullfile(baseDir,'QRGMMoutputtest2'), ...
    fullfile(baseDir,'savetime'), ...
    fullfile(baseDir,'ECPX') ...
    };

for k = 1:numel(dirList)
    if ~exist(dirList{k}, 'dir')
        mkdir(dirList{k});
    end
end

%% Generate data using QRGMM
run=100;
noutput=100000;

onlinetime_QRGMM_x_1=zeros(run,1);
onlinetime_QRGMM_x_2=zeros(run,1);

x=[70,0.05,0.2,0.8];
px=[1,Poly(x,2)];

for runi = 1:run
    rng(runi)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% QRGMM %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    global le;
    global ue;
    global quantilepoints
    global coefficients
    m=300;
    le=1/m;
    ue=1-le;
    quantilenum=m;
    quantilepoints=linspace(le,ue,m-1);
    
    %%%% Load Quantile Regression Coefficients for Y1 (Learned via R) %%%%%

    filename=strcat('./data/QRGMMcoeff1/QRGMMcoeff1_',num2str(runi),'.csv');
    coefficients=csvread(filename,1,1);

    %%%%%%%%%%%%%%%%%%% Generate Conditional Test Data  %%%%%%%%%%%%%%%%%%%

    tic
    QRGMMoutput1_1=QRGMM_xstar_fun(noutput,px);
    onlinetime_QRGMM_x_1(runi)=toc;

    filename=strcat('./data/QRGMMoutput1_1/QRGMMoutput1_1_',num2str(runi),'.csv');
    csvwrite(filename,QRGMMoutput1_1);

    %%%%%%%%%%%%%%%%%% Generate Unconditional Test Data %%%%%%%%%%%%%%%%%%%

    filename=strcat('./data/ECtestdata1/ECtestdata1_',num2str(runi),'.csv');
    QRGMMtestdata1=csvread(filename,0,0);

    QRGMMoutputtest1=QRGMM_fun(length(QRGMMtestdata1),QRGMMtestdata1(:,1:size(px,2)));

    filename=strcat('./data/QRGMMoutputtest1/QRGMMoutputtest1_',num2str(runi),'.csv');
    csvwrite(filename,QRGMMoutputtest1);

    %%%% Load Quantile Regression Coefficients for Y2 (Learned via R) %%%%%

    filename=strcat('./data/QRGMMcoeff2/QRGMMcoeff2_',num2str(runi),'.csv');
    coefficients=csvread(filename,1,1);

    %%%%%%%%%%%%%%%%%%% Generate Conditional Test Data  %%%%%%%%%%%%%%%%%%%
 
    tic
    QRGMMoutput1_2=QRGMM_xstar_fun(noutput,px);
    onlinetime_QRGMM_x_2(runi)=toc;

    filename=strcat('./data/QRGMMoutput1_2/QRGMMoutput1_2_',num2str(runi),'.csv');
    csvwrite(filename,QRGMMoutput1_2);

    %%%%%%%%%%%%%%%%%% Generate Unconditional Test Data %%%%%%%%%%%%%%%%%%%

    filename=strcat('./data/ECtestdata2/ECtestdata2_',num2str(runi),'.csv');
    QRGMMtestdata2=csvread(filename,0,0);

    QRGMMoutputtest2=QRGMM_fun(length(QRGMMtestdata2),QRGMMtestdata2(:,1:size(px,2)));

    filename=strcat('./data/QRGMMoutputtest2/QRGMMoutputtest2_',num2str(runi),'.csv');
    csvwrite(filename,QRGMMoutputtest2);

end

savetime=[onlinetime_QRGMM_x_1,onlinetime_QRGMM_x_2];
filename='./data/savetime/onlinetime_QRGMM.csv';
csvwrite(filename,savetime);

filename='./data/ECPX/ECPX.csv';
csvwrite(filename,px);
