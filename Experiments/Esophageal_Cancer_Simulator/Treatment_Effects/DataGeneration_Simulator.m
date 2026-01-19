clear
clc

%% ---- Create all output folders once (before the loop) ----
baseDir = './data';

dirList = { ...
    fullfile(baseDir,'ECtraindata1'), ...
    fullfile(baseDir,'ECtraindata2'), ...
    fullfile(baseDir,'traindata1'), ...
    fullfile(baseDir,'traindata2'), ...
    fullfile(baseDir,'ECtestdata1'), ...
    fullfile(baseDir,'ECtestdata2'), ...
    fullfile(baseDir,'testdata1'), ...
    fullfile(baseDir,'testdata2'), ...
    fullfile(baseDir,'Systemoutput1_1'), ...
    fullfile(baseDir,'Systemoutput1_2'), ...
    fullfile(baseDir,'testdf1_1'), ...
    fullfile(baseDir,'testdf1_2') ...
    };

for k = 1:numel(dirList)
    if ~exist(dirList{k}, 'dir')
        mkdir(dirList{k});
    end
end


%% Generate data using simulation model

run=100;
x=[70,0.05,0.2,0.8];
nsample=10000;
noutput=100000;


for runi = 1:run
    rng(runi)
    
    %%%%%%%%%%%%%%%%%%%%%%%%%% Train Data  %%%%%%%%%%%%%%%%%%%%%%%%
    
    X1=randi([55 80],nsample,1);
    X2=0.1*rand(nsample,1);
    X3=rand(nsample,1);
    X4=rand(nsample,1);

    X=[X1 X2 X3 X4];
    PX=Poly(X,2);
    repN=1;
    GMMtraindata1=zeros(size(PX,1)*repN,size(PX,2)+2);
    GMMtraindata1(:,2:size(PX,2)+1)=repelem(PX,repN,1);
    GMMtraindata1(:,1)=1;

    GMMtraindata2=zeros(size(PX,1)*repN,size(PX,2)+2);
    GMMtraindata2(:,2:size(PX,2)+1)=repelem(PX,repN,1);
    GMMtraindata2(:,1)=1;

    for i=1:size(PX,1)
        GMMtraindata1((i-1)*repN+1:i*repN,end) = EsophagealCancerSim(X(i,2),X(i,3),X(i,4),1,X(i,1),repN,0,'raw');
    end

    for i=1:size(PX,1)
        GMMtraindata2((i-1)*repN+1:i*repN,end) = EsophagealCancerSim(X(i,2),X(i,3),X(i,4),2,X(i,1),repN,0,'raw');
    end

    filename1=strcat('./data/ECtraindata1/ECtraindata1_',num2str(runi),'.csv');
    csvwrite(filename1,GMMtraindata1);

    filename2=strcat('./data/ECtraindata2/ECtraindata2_',num2str(runi),'.csv');
    csvwrite(filename2,GMMtraindata2);
 
    traindata1=[GMMtraindata1(:,2:5),GMMtraindata1(:,end)];
    filename1=strcat('./data/traindata1/traindata1_',num2str(runi),'.csv');
    csvwrite(filename1,traindata1);  

    traindata2=[GMMtraindata2(:,2:5),GMMtraindata2(:,end)];
    filename2=strcat('./data/traindata2/traindata2_',num2str(runi),'.csv');
    csvwrite(filename2,traindata2);  
 
    %%%%%%%%%%%%%%%%%%%%%%% Unconditional Test Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Xtest1=randi([55 80],nsample,1);
    Xtest2=0.1*rand(nsample,1);
    Xtest3=rand(nsample,1);
    Xtest4=rand(nsample,1);

    Xtest=[Xtest1 Xtest2 Xtest3 Xtest4];
    PXtest=Poly(Xtest,2);
    repN=1;

    GMMtestdata1=zeros(size(PXtest,1)*repN,size(PXtest,2)+2);
    GMMtestdata1(:,2:size(PXtest,2)+1)=repelem(PXtest,repN,1);
    GMMtestdata1(:,1)=1;

    GMMtestdata2=zeros(size(PXtest,1)*repN,size(PXtest,2)+2);
    GMMtestdata2(:,2:size(PXtest,2)+1)=repelem(PXtest,repN,1);
    GMMtestdata2(:,1)=1;

    for i=1:size(PXtest,1)
        GMMtestdata1((i-1)*repN+1:i*repN,end) = EsophagealCancerSim(Xtest(i,2),Xtest(i,3),Xtest(i,4),1,Xtest(i,1),repN,0,'raw');
    end

    filename1=strcat('./data/ECtestdata1/ECtestdata1_',num2str(runi),'.csv');
    csvwrite(filename1,GMMtestdata1);

    for i=1:size(PXtest,1)
        GMMtestdata2((i-1)*repN+1:i*repN,end) = EsophagealCancerSim(Xtest(i,2),Xtest(i,3),Xtest(i,4),2,Xtest(i,1),repN,0,'raw');
    end

    filename2=strcat('./data/ECtestdata2/ECtestdata2_',num2str(runi),'.csv');
    csvwrite(filename2,GMMtestdata2);

    testdata1=[GMMtestdata1(:,2:5),GMMtestdata1(:,end)];
    filename1=strcat('./data/testdata1/testdata1_',num2str(runi),'.csv');
    csvwrite(filename1,testdata1); 

    testdata2=[GMMtestdata2(:,2:5),GMMtestdata2(:,end)];
    filename2=strcat('./data/testdata2/testdata2_',num2str(runi),'.csv');
    csvwrite(filename2,testdata2);  

    %%%%%%%%%%%%%%%%%%%%%%%%%% Conditional Test Data  %%%%%%%%%%%%%%%%%%%%%%

    rng(runi)

    Systemoutput1_1=EsophagealCancerSim(x(2),x(3),x(4),1,x(1),noutput,0,'raw');

    Systemoutput1_2=EsophagealCancerSim(x(2),x(3),x(4),2,x(1),noutput,0,'raw');

    filename1=strcat('./data/Systemoutput1_1/Systemoutput1_1_',num2str(runi),'.csv');
    csvwrite(filename1,Systemoutput1_1);

    filename2=strcat('./data/Systemoutput1_2/Systemoutput1_2_',num2str(runi),'.csv');
    csvwrite(filename2,Systemoutput1_2);
    
    testdf1_1=zeros(noutput,5);
    testdf1_1(:,1:4)=repelem(x,noutput,1);
    testdf1_1(:,5)=Systemoutput1_1';
    filename1=strcat('./data/testdf1_1/testdf1_1_',num2str(runi),'.csv');
    csvwrite(filename1,testdf1_1); 

    testdf1_2=zeros(noutput,5);
    testdf1_2(:,1:4)=repelem(x,noutput,1);
    testdf1_2(:,5)=Systemoutput1_2';
    filename2=strcat('./data/testdf1_2/testdf1_2_',num2str(runi),'.csv');
    csvwrite(filename2,testdf1_2);

end