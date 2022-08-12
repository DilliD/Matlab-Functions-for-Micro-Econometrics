function hess=hessp(a,qi,h)
% hess=hessp(a,qi,h)
% Procedure for Calculating Numerical Hessian
if nargin<3; h=[0.00001,0.00001]; end
n=size(qi(a),1); k=size(a,1); eyek=eye(k); h=0.0001; hess=zeros(k,k);
for j1=1:k
     for j2=1:k
        hj1=h*eyek(:,j1); hj2=h*eyek(:,j2);
        upper=sum(qi(a+hj1+hj2))-sum(qi(a+hj1-hj2));
        lower=sum(qi(a-hj1+hj2))-sum(qi(a-hj1-hj2));
        hess(j1,j2)=(upper-lower)./(4*h*h);
    end
end
end