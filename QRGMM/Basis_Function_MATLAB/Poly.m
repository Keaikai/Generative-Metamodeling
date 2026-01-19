function [PX] = Poly(X,degree)

n=size(X,1);
d=size(X,2);

if degree==2
    %pd=d+d+nchoosek(d,2); 
    PX=[X,X.^2];
    ind=1:d;
    cind2=nchoosek(ind,2);
    for i =1:nchoosek(d,2)
        PX=[PX,X(:,cind2(i,1)).*X(:,cind2(i,2))];
    end
end

% if degree==2
%     pd=d+d+nchoosek(d,degree); 
%     PX=zeros(n,pd);
%     for i=1:n
%         interprodx=nchoosek(X(i,:),2);
%         interprodx2=interprodx(:,1).*interprodx(:,2);
%         PX(i,:)=[X(i,:),X(i,:).^2,interprodx2'];
%     end
% end

if degree==3
    %pd=d+d+d+nchoosek(d,2)+nchoosek(d,3); 
    PX=[X,X.^2,X.^3];
    ind=1:d;
    cind2=nchoosek(ind,2);
    for i =1:nchoosek(d,2)
        PX=[PX,X(:,cind2(i,1)).*X(:,cind2(i,2))];
    end
    cind3=nchoosek(ind,3);
    for i =1:nchoosek(d,3)
        PX=[PX,X(:,cind3(i,1)).*X(:,cind3(i,2)).*X(:,cind3(i,3))];
    end    
end

% if degree==3
%     pd=d+d+d+nchoosek(d,2)+nchoosek(d,3); 
%     PX=zeros(n,pd);
%     for i=1:n
%         interprodx20=nchoosek(X(i,:),2);
%         interprodx21=interprodx20(:,1).*interprodx20(:,2);
%         interprodx30=nchoosek(X(i,:),3);
%         interprodx31=interprodx30(:,1).*interprodx30(:,2).*interprodx30(:,3);
%         PX(i,:)=[X(i,:),X(i,:).^2,X(i,:).^3,interprodx21',interprodx31'];
%     end
% end

end
