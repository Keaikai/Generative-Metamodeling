function y = EsophagealCancerSim(risk,aspirinEffect,statinEffect,...
                                  drugIndex,initialAge,repN,CV,return_type)
% Esophageal Cancer Simulator, adopted from
% [1] Hur, C., N. S. Nishioka, and G. S. Gazelle (2004). Cost-effectiveness of 
% aspirin chemoprevention for Barrett's esophagus. J. Natl. Canc. Inst. 96 (4), 316-325.
% [2] Choi, S. E., K. E. Perzan, A. C. Tramontano, C. Y. Kong, and C. Hur (2014).
% Statins and aspirin for chemoprevention in Barrett's esophagus: Results of a
% cost-effectiveness analysis. Canc. Prev. Res. 7 (3), 341-350.
%
% -- Input --
% risk: annual Barrett's esophagus to cancer probability, [0,0.1]
% aspirinEffect: [0,1], new risk = risk x (1 - aspirinEffect)
% statinEffect: [0,1], new risk = risk x (1 - statinEffect)
% drugIndex: 0 - use no drug; 1 - use aspirin; 2 - use statin
% initialAge: the age that the simulated patient starts with, {55,56,...,80}
% repN: replication number for simulation
% CV: 1 - use control variate (CV) to reduce variance; 0 - do not use CV
% return_type: see description below
% -- Output --
% if return_type == []
% y: simulated quality-adjusted life years (QALY), sample mean
% if return_type == "mean_var"
% [y, yvar] yvar: sample variance, not variance of y
% if return_type == "raw"
% y: simulated quality-adjusted life years (QALY), all sample points
%
% -- Example --
% y = EsophagealCancerSim(0.08,0.4,0.2,0,55,10000,0)
% y = EsophagealCancerSim(0.08,0.4,0.2,1,55,10000,0,'mean_var')
% y = EsophagealCancerSim(0.08,0.4,0.2,1,55,10000,1,'mean_var')
% y = EsophagealCancerSim(0.08,0.4,0.2,2,55,10000,1,'raw')
%
% -- Warning --
% due to high simulation variance, to get high accurate result, repN has to
% be very large, e.g., 500,000, even with CV
%
% Implemented by SHEN, Haihui, shenhaihui@gmail.com
% Last update: June 26, 2020


p1 = 1 - (1 - risk)^(1/12);
annual_esophageal_cancer_mortality = 0.29;
p3 = 1 - (1 - annual_esophageal_cancer_mortality)^(1/12);

if drugIndex ~= 0 % use aspirin or statin
    annual_complication = [0.0024, 0.001];
    drug_factor = [1 - aspirinEffect, 1 - statinEffect];
    drug_comp_cure = [0.9576, 0.998];

    k = risk * drug_factor(drugIndex) / annual_complication(drugIndex);
    p4 = (1 - (1 - risk * drug_factor(drugIndex) - annual_complication(drugIndex))^(1/12)) / (1+k);
    p11 = k * p4; % p1'
    p5 = drug_comp_cure(drugIndex);   
else % no drug   
    p11 = 0;  % these states will never be entered
    p4 = 0;
    p5 = 0;
end

% age-related all-cause mortality (natural mortality) probability, for 55 
% years old, 56, ..., 100
% data source: Arias, Elizabeth (2015). United States Life Tables, 2011. 
% National Vital Statistics Reports. 64 (11). Available from:
% https://www.cdc.gov/nchs/data/nvsr/nvsr64/nvsr64_11.pdf
dyingP_55_100 = [0.007779, 0.008415, 0.009074, 0.009727, 0.010371, ...
                 0.011034, 0.011738, 0.012489, 0.013335, 0.014319, ...
                 0.015482, 0.016824, 0.018330, 0.019900, 0.021539, ...
                 0.023396, 0.025476, 0.027794, 0.030350, 0.033204, ...
                 0.036345, 0.039788, 0.043720, 0.048335, 0.053650, ...
                 0.059565, 0.065848, 0.072956, 0.080741, 0.089357, ...
                 0.099650, 0.110901, 0.123146, 0.136412, 0.150710, ...
                 0.166038, 0.182374, 0.199676, 0.217880, 0.236903, ...
                 0.256636, 0.276954, 0.297713, 0.318755, 0.339914, ...
                 1.000000];
dyingP_1_100 = [zeros(1,54), dyingP_55_100];    

% transition probability matrix (without p2)
P = [1-p1, p1,  0,  0,    0,    0,    0,        0;   ...
     0,    0,   0,  0,    0,    0,    0,        0;   ...
     0,    0,   0,  0.8,  0.16, 0.04, 0,        0;   ...
     0,    0,   0,  1,    0,    0,    0,        0;   ...
     0,    0,   0,  0,    1-p3, p3,   0,        0;   ...
     0,    0,   0,  0,    0,    1,    0,        0;   ...
     0,    p11, 0,  0,    0,    0,    1-p11-p4, p4;  ...
     p5,   0,   0,  0,    0,    1-p5, 0,        0];   

Y = zeros(1,repN);
if CV == 1
    X = zeros(1,repN);
end

for rep = 1:repN
% generate the RV for age-related mortality, at the beginning
nat_mor_rand = rand(1,100);
nat_mor_flag = 0;

state = zeros(1,540);
i = 1;
if drugIndex ~= 0 % use aspirin or statin
    state(i) = 7;
else
    state(i) = 1;
end    

age = initialAge - 1; 
while 1
    age = age + 1;    
    p2 = -0.0023 * age + 1.1035;
    P(2,3) = p2; % update transition probability matrix
    P(2,4) = 1-p2;

    for month = 1:6
        PNext = P(state(i),:);
        PNextCum = cumsum(PNext);    
        i = i + 1;
        state(i) = find(PNextCum>=rand,1);
    end    
    
    % age-related mortality (assume it happens in the middle of year)
    if nat_mor_rand(age) <= dyingP_1_100(age)
        nat_mor_flag = 1;
        break;
    end
    
    for month = 7:12
        PNext = P(state(i),:);
        PNextCum = cumsum(PNext);    
        i = i + 1;
        state(i) = find(PNextCum>=rand,1);
    end
    
    if state(i) == 6  % death
        break;
    end
end

if state(i) == 6
    lasti = find(state == 6,1) - 1; % patient may be dead earlier before
else
    lasti = i - 1;
end
state = state(1:lasti);

% quality-of-life adjustment
lifelength = ones(1,lasti);
lifelength(state==2) = 0.5;
lifelength(state==3) = 0.5;
lifelength(state==5) = 0.5;
lifelength(state==4) = 0.97;

% aspirin && complication && cured, may cause disability, which affects the
% life quality for the rest of life
if (drugIndex==1) && (~isempty(find(state==8,1))) &&  (lasti > find(state==8,1))
    if rand <= 0.058
        disability_i = find(state==8,1);
        lifelength(disability_i+1:lasti) = lifelength(disability_i+1:lasti) * 0.61;
    end  
end

QALY = sum(lifelength) / 12;
Y(rep) = QALY;

if CV == 1
    %%% control variate
    if nat_mor_flag == 0 % death due to cancer
        % what if there is no cancer?
        index = find(nat_mor_rand(initialAge:100) <= dyingP_1_100(initialAge:100),1);
        control_variate = index - 0.5;
    else                 % death due to natural reason
        control_variate = lasti / 12;
    end
    X(rep) = control_variate;
end
end

if CV == 1
    %%% use control variate to reduce variance:  Y - b( CV - ECV)
    %%% control variate, CV, the natural life length (after 55)
    % expected natural life
    reage = 100 - initialAge + 1;
    P = zeros(reage,1);
    for i = 1:reage
        P(i) = dyingP_1_100(initialAge-1+i);
        for j = 1:i-1
            P(i) = P(i)*(1-dyingP_1_100(initialAge-1+j));
        end        
    end
    ECV = ((1:reage)-0.5) * P; % when initialAge=55, ECV = 
    b = 0.9; % the empirical optimal value for control variate

    Y = Y - b * (X - ECV);
end

if nargin == 7
    y = mean(Y);
elseif nargin == 8 
    if return_type == "mean_var"
        y = [mean(Y), var(Y)];
    elseif return_type == "raw"
        y = Y;
    end
end