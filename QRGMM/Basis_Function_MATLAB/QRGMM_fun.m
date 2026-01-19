function [output] = QRGMM_fun(output_size,x_mat)

output=zeros(output_size,1);
u=rand(output_size,1);

global le;
global ue;
global quantilepoints
global coefficients

newcoefficients=[coefficients(1,:);coefficients; coefficients(end,:)];
order=u/le;

location=floor(order);
alpha=order-floor(order);

b1=newcoefficients(location+1,:);
b2=newcoefficients(location+2,:);

b=b1.*(1-alpha)+b2.*(alpha);

output=sum(b.*x_mat,2);

end