function stat=clogit(y,x,print, step, iterlim, algorithm)
% stat=clogit(y,x,print, step, iterlim, algorithm)
%
% Version 1.0, MA(2022.4.16)
% Editor : TaeGyu, Yang, MA of Economics, Korea University
%
% This procedure aims to calculate nonlinear panel conditional logit estimates.
% Input:
% y : dependent variable
% x : matrix of time-varying variables
% print : input "print" if wanting displaying result table.
% step : hyper-parameter affecting on convergence speed of BHHH algorithm
% iterlim : upper limit of iternation
% algorithm : Choose which algorithm would be used for numerical optimization
%     if algorithm="fminsearch" : Use Matlab fminsearch
%     if algorithm="BHHH" : Berndt-Hall-Hall-Hausman Algorithm
%     if algorithm="NR"    : Newton-Raphson Algorithm
%     if algorithm="GD"    : Gradient Descent Algorithm
%
% Output :
% stat is a structure object
% stat.para : estimated parameter vector
% stat.information : fisher information matrix
% stat.vcov : variance-covariance matrix
% stat.se : standard error for each parameter
% stat.tv : t-value for each parameter
% stat.pv : p-value
% stat.score : Estimated score matrix
% stat.influ : influence function
% stat.max : Maximund of likelihood function
if nargin==2; print=""; step=0.5; iterlim=500; algorithm="BHHH";
elseif nargin==3; step=0.5; iterlim=500; algorithm="BHHH";
elseif nargin==4; iterlim=500; algorithm="BHHH";
elseif nargin==5; algorithm="BHHH"; end
if isstring(print)~=1; print=string(print); end
if isstring(algorithm)~=1; algorithm=string(algorithm); end
T=size(y,2);
if iscell(y)==1; for iter=1:T; y0(:,iter)=y{iter}; end; clearvars y; y=y0; end
if iscell(x)==1; for iter=1:T; x0(:,:,iter)=x{iter}; end; clearvars x; x=x0; end
kx=size(x,2); n=size(x,1); G=GenGrid(T);
initial=0.5*ones(kx+T-1,1);
qi=@(para)(like(y,x,G, para));
if algorithm=="BHHH"
    result=BHHH(qi, initial, step, print, iterlim);
elseif algorithm=="NR"
    result=NR(qi, initial, step, print, iterlim);
elseif algorithm=="GD"
    result=grd_desc(qi, initial, step, print, iterlim);
end
stat.converge=result.converge;
if stat.converge==1
    stat.para=result.best;
    score=gradp(stat.para, qi);
    stat.information=score'*score;
    stat.vcov=inv(stat.information);
    stat.se=sqrt(diag(stat.vcov));
    stat.tv=stat.para./stat.se;
    stat.pv=2*tcdf(abs(stat.tv), n*T-size(stat.para,1), 'upper');
    stat.score=score;
    stat.influ=score*(stat.vcov);
    stat.max=sum(qi(stat.para));
else
    disp("Optimization did not converge");
end
if print=="print"; ShowTable(stat, T, kx, n); end
end

function result=ShowTable(stat, T, kx, n)
result=[round(stat.para,3), round([stat.se, stat.tv, stat.pv],2)];
hd=["t"+num2str((2:T)')+" - t"+num2str((1:T-1)'); "b"+num2str((1:kx)')];
result=[hd, result];
disp("=============================================")
disp("                  <Conditional Logit>")
disp(" ")
disp("                                          Wave : "+num2str(T))
disp("                                  Sample Size : "+num2str(n))
disp("---------------------------------------------")
disp("  Parameter  |   Est.  |  Std.Err. |   tv.  |  pv.  ")
disp("---------------------------------------------")
disp(result);
disp("---------------------------------------------")
disp("maximand = "+num2str(stat.max))
disp("=============================================")
end

function q=like(y, x, G, para)
[n,k]=size(x); T=size(y,2);
t0=[0;para(1:T-1,:)]; b0=para(T:end,:);
for iter=1:T
    xb(:,iter)=t0(iter,1)+x(:,:,iter)*b0;
    dxb(:,iter)=xb(:,iter)-xb(:,1);
end
yxb=y.*dxb; num=exp(sum(yxb')'); q=0;
for iter=1:T-1
    d=sum(y')'==iter; g0=G{iter}; ng0=size(g0,1); exb=0;
    for iter2=1:ng0
        gxb=xb.*g0(iter2,:); exb=exb+exp(sum(gxb')');
    end
    q=q+d.*log( eps + (num./exb) );
end
end

function G=GenGrid(T)
v=(1:T)';
for iter1=1:T-1
    idx=nchoosek(v,iter1); nidx=size(idx,1); lamb=zeros(nidx,T);
    for iter2=1:nidx
        lamb(iter2,idx(iter2,:)')=1;
    end
    lamb=lamb-iter1*[ones(nidx,1), zeros(nidx,T-1)]; G{iter1}=lamb;
end
end