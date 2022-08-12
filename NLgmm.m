function stat=NLgmm(mc, initial, step, print, iterlim, header)
% stat=NLgmm(mc, initial, step, print, iterlim, header)
%
% Version 1.0 (2022.4.10)
% Editor : TaeGyu, Yang, MA of Economics, Korea University
%
% This procedure provides linear/nonlinear GMM estimator
%
% Input:
% mc : moment condition, you should exploit symbolic expression
% initial : initial point of parameter input
% step : hyper-parameter affecting on convergence speed of BHHH algorithm
% print : if print="print"; print iteration process and estimation result table
% iterlim : upper limit of iternation
% header: Variable Name
%
% Output :
% stat is a structure object
% stat.para : estimated parameter vector
% stat.vcov : variance-covariance matrix
% stat.se : standard error for each parameter
% stat.tv : t-value for each parameter
% stat.pv : p-value
% stat.influ : influence function
% stat.overid : overidentification test statistics and p-value
if nargin==2; step=0.5; print=""; iterlim=200; header=[];
elseif nargin==3; print=""; iterlim=200; header=[];
elseif nargin==4; iterlim=200; header=[]; 
elseif nargin==5; header=[]; end
if isstring(print)~=1; print=string(print); end
if isstring(header)~=1; header=string(header); end
if size(header,2)>size(header,1); header=header'; end

k=size(initial,1); [n,q]=size(mc(initial));
sum_mc=@(b)( sum( mc(b) )' );
obj_1st=@(b)( sum_mc(b)'*sum_mc(b) );
obj_2nd=@(b)( sum_mc(b)'*inv(mc(b)'*mc(b))*sum_mc(b) );

disp("1st Stage Estimation");
[bive,converge]=optim(mc, sum_mc, obj_1st, eye(q), initial, step, iterlim);
m0=mc(bive); invmm=inv(m0'*m0);
disp("2nd Stage Estimation");
[bgmm,converge]=optim(mc, sum_mc, obj_2nd, invmm, bive, step, iterlim);
gr=gradp(bgmm, sum_mc); vcov=inv(gr'*invmm*gr);

stat.para=bgmm;  stat.vcov=vcov;
stat.se=sqrt(diag(vcov)); stat.tv=bgmm./sqrt(diag(vcov));
stat.pv=2*tcdf(abs(stat.tv),n-k,'upper');
stat.influ=mc(bgmm)*invmm*gr*vcov;
if q>k; stat.overid=[obj_2nd(bgmm), 1-chi2cdf(obj_2nd(bgmm),q-k)]; end
if print=="print"; showtable(stat, header, n, q, k); end
end

function [b, converge]=optim(mc, sum_mc, obj, invmm, initial, step, iterlim)
niter=1; b0=initial; bestobj=obj(b0); converge=0;
while niter<iterlim
gr=gradp(b0, sum_mc); % q by k
b1 = b0 - step*inv(gr'*invmm*gr)*gr'*invmm*sum_mc(b0);
newobj=obj(b1);
disp(num2str([niter, newobj, bestobj]));
if abs(newobj-bestobj)<0.0001; disp("Iteration Converges"); converge=1; break;  end
if bestobj>newobj; bestobj=newobj; end
b0=b1; niter=niter+1;
end
b=b1;
end

function result=showtable(stat,header, n, q, k)
if size(header,2)>size(header,1); header=header'; end
if size(header,1)~=k; header="x"+num2str((1:k)'); end
result=[round(stat.para,3), round([stat.se, stat.tv, stat.pv],2)];
disp("==========================================")
disp("                           <Non-Linear GMM>")
disp("                                                    sample size = " + num2str(n));
disp("------------------------------------------")
disp("   Para   |    Est.   |   Std.Err.  |   T-value  |  p-value ")
disp("------------------------------------------")
disp([header, result]);
if q>k
    disp("------------------------------------------")
    disp("Over-Identification Test")
    disp(" Jn = "+num2str(round(stat.overid(1,1),2))+",  Pr(X > Jn) = "+num2str(round(stat.overid(1,2),2)) );
end
disp("==========================================")
end