function result=BHHH(qi, initial, step, print, iterlim, h)
% result=BHHH(qi, initial, step, print, iterlim, h)
% This procedure(Berndt-Hall-Hall-Hausman Algorithm) is a solver of maximization probelm :
%
% Input:
% qi : objective function for every observation
% initial : initial point of parameter input
% step : hyper-parameter affecting on convergence speed of algorithm
% print : if print="print"; print iteration process and estimation result table
% iterlim : upper limit of iternation
% h : error rate in terminal step
%
% Output :
% result is a structure object
% result.best : MDE estimate vector
% result.max : maximum value of objective function
% result.converge : 1 if the algorithm converges
result.best=a1; result.max=sum(qi(a1)); result.converge=converge;
if nargin==2; step=0.2; print=""; iterlim=200; h=0.0001;
elseif nargin==3; print=""; iterlim=200; h=0.0001;
elseif nargin==4; iterlim=200; h=0.0001;
elseif nargin==5; h=0.0001;
end
a0=initial; niter=0; converge=0; bestobj=sum(qi(a0));
while niter<iterlim
    gradi=gradp(a0,qi);
    M=gradi'*gradi; M=0.5*(M+M'); %invM=inv(M);
    invM=pinv(M+0.001*eye(size(M,1)));
    a1=a0+step*(invM)*sum(gradi)';
    newobj=sum(qi(a1)); niter=niter+1;
    if print=="print"; disp(num2str([niter,newobj,bestobj])); end
    if abs(newobj-bestobj)<0.0001; converge=1; break
    elseif newobj>bestobj; bestobj=newobj;
    end
    a0=a1;
end
result.best=a1; result.max=sum(qi(a1)); result.converge=converge;
end