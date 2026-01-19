function [choice,sample_size] = KN_simulator(Xsample,k,alpha,delta,n0)
T = size(Xsample,1);
choice = zeros(T,1);
sample_size = zeros(T,1);

%Step 1: Setup
yita = ((2*alpha/(k-1))^(-2/(n0-1))-1)/2;    

for t = 1:T
    %Step 2: Initialization
    Alternative= 1:k;
    I = Alternative;
    h_square = 2*yita*(n0-1);
    ini_output = zeros(k,n0);
    x_bar = zeros(1,k);
    prime_sample_size = zeros(1,k);
    S = zeros(k,k); 
    for i = 1:k
        ini_output(i,:) = EsophagealCancerSim(Xsample(t,2),Xsample(t,3),Xsample(t,4),i,Xsample(t,1),n0,0,'raw');
        x_bar(1,i) = sum(ini_output(i,:))/n0;
    end
    for i = 1:k
        for l = 1:k
            if i~=l
                S(i,l) = var(ini_output(i,:)-ini_output(l,:));
            end
        end
    end
    r = n0;
    %Step 3: Screening
    while 1
        I_old = I;
        num = length(I_old);
        W = zeros(num,num);
        hope = I_old;
        for i = I_old
            for l = I_old
                if i~=l
                    W(i,l) = max(0,h_square*S(i,l)/(2*r*delta)-delta/2);
                    if x_bar(i) < x_bar(l)-W(i,l)
                        hope(hope==i) = [];
                        prime_sample_size(i)=r;
                        break;
                    end
                end
            end
        end
        I = hope;
        if length(I) == 1
             break; 
        end
        r = r+1;
        for i = I
            x_bar(1,i) = ((n0+r-1)*x_bar(1,i)+EsophagealCancerSim(Xsample(t,2),Xsample(t,3),Xsample(t,4),i,Xsample(t,1),1,0,'raw'))/(n0+r);
        end
    end
    prime_sample_size(I)=r;
    choice(t) = I;
    sample_size(t) = sum(prime_sample_size);
end
end