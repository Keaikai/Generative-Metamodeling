function output = QRGMM_xstar_fun(output_size,x)

output=zeros(output_size,1);
u=rand(output_size,1);

global le;
global ue;
global quantilepoints
global coefficients

coefficients=[coefficients(1,:);coefficients; coefficients(end,:)];
quantilecurve=sum(coefficients.*x,2);


order=u/le;
location=floor(order);
alpha=order-floor(order);
q1=quantilecurve(location+1);
q2=quantilecurve(location+2);
q=q1.*(1-alpha)+q2.*(alpha);
output=q;

end