function [output] = QRGMM_fun_Y2(output_size,x_mat)

output=zeros(output_size,1);
u=rand(output_size,1);

global le;
global ue;
global quantilepoints
global coefficients2

coefficients2=[coefficients2(1,:);coefficients2; coefficients2(end,:)];
order=u/le;

location=floor(order);
alpha=order-floor(order);

b1=coefficients2(location+1,:);
b2=coefficients2(location+2,:);

b=b1.*(1-alpha)+b2.*(alpha);

output=sum(b.*x_mat,2);

end