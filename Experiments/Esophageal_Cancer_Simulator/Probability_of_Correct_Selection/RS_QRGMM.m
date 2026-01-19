function choice = RS_QRGMM(Xsample)

T = size(Xsample,1);
choice = zeros(T,1);
noutput=100000;

PX=[ones(T,1) Poly(Xsample,2)];
%PX=[ones(T,1) Xsample];

for t=1:T
    px_mat = repmat(PX(t,:),noutput,1);
    output1 = QRGMM_fun_Y1(noutput,px_mat);
    output2 = QRGMM_fun_Y2(noutput,px_mat);
    choice(t) = (mean(output1)<=mean(output2))+1;
end
end