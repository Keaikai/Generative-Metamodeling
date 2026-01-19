function [output] = QRGMM_fun_Y1(output_size,x_mat)

output=zeros(output_size,1);
u=rand(output_size,1);

global le;
global ue;
global quantilepoints
global coefficients1

coefficients1=[coefficients1(1,:);coefficients1; coefficients1(end,:)];
order=u/le;

location=floor(order);
alpha=order-floor(order);

b1=coefficients1(location+1,:);
b2=coefficients1(location+2,:);

b=b1.*(1-alpha)+b2.*(alpha);

output=sum(b.*x_mat,2);

end