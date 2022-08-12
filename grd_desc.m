function result=grd_desc(qi, initial, step, print, iterlim, fix)
% result=grd_desc(qi, initial, step, print, iterlim, h)
% This procedure(Gradient Descent Algorithm) is a solver of maximization probelm :
% See Barzilai, Jonathan(1988)
% Input:
% qi : objective function for every observation
% initial : initial point of parameter input
% step : hyper-parameter affecting on convergence speed of algorithm
% print : if print="print"; print iteration process and estimation result table
% iterlim : upper limit of iternation
% fix : if fix="fix", convergence rate for every step is being fixed
%
% Output :
% result is a structure object
% result.best : MDE estimate vector
% result.max : maximum value of objective function
% result.converge : 1 if the algorithm converges
if nargin==5; fix=""; end
a0=initial; niter=0; converge=0; bestobj=sum(qi(a0));
if fix=="fix"; gn=update_gn(a0, a0+(1.5*rand(size(a0))), qi); else; gn=1; end
while niter<iterlim
    gradi=gradp(a0,qi);
    a1=a0+step*gn*sum(gradi)';
    newobj=sum(qi(a1)); niter=niter+1;
    if (print=="print")+(print=="plot")==1; disp(num2str([niter,newobj,bestobj])); end
    if abs(newobj-bestobj)<0.0001; converge=1; break
    elseif newobj>bestobj; bestobj=newobj;
    end
    if fix=="fix"; gn=update_gn(a0, a1, qi); else; gn=1; end
    a0=a1;
end
result.best=a1; result.max=sum(qi(a1)); result.converge=converge;
end

function g=update_gn(a0, a1, qi)
gr1=sum(gradp(a1,qi))'; gr0=sum(gradp(a0,qi))';
g=abs((a1-a0)'*(gr1-gr0))/(norm(gr1-gr0))^2;
end