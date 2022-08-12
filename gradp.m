function gr=gradp(a,qi,h)
% gr=gradp(a,qi,h)
% Procedure for Calculating Numerical Gradient
if nargin<3; h=0.00001; end
n=size(qi(a),1); k=size(a,1); eyek=eye(k); gr=zeros(n,k);
for j=1:k; hj=h*eyek(:,j); gr(:,j)=(qi(a+hj)-qi(a-hj))./(2*h); end
end