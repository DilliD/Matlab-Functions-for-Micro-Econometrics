function result=NR(qi, initial, step, print, iterlim)
% result=NR(qi, initial, step, print, iterlim, h)
% This procedure(Newton-Raphson Algorithm) is a solver of maximization probelm :
%
% Input:
% qi : objective function for every observation
% initial : initial point of parameter input
% step : hyper-parameter affecting on convergence speed of algorithm
% print : if print="print"; print iteration process and estimation result table
% iterlim : upper limit of iternation
%
% Output :
% result is a structure object
% result.best : MDE estimate vector
% result.max : maximum value of objective function
% result.converge : 1 if the algorithm converges
a0=initial; niter=0; converge=0; bestobj=sum(qi(a0));
while niter<iterlim
    gradi=gradp(a0,qi);
    H=hessp(a0,qi); H=0.5*(H+H'); invH=inv(H); a1=a0+step*(-invH)*sum(gradi)';
    newobj=sum(qi(a1)); niter=niter+1;
    if (print=="print")+(print=="plot")==1; disp(num2str([niter,newobj,bestobj])); end
    if abs(newobj-bestobj)<0.0001; converge=1; break
    elseif newobj>bestobj; bestobj=newobj;
    end
    a0=a1;
end
result.best=a1; result.max=sum(qi(a1)); result.converge=converge;
end