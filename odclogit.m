function stat=odclogit(y, x, print, step, iterlim, algorithm)
% stat=odclogit(y, x, print, step, iterlim, algorithm)
%
% Version 1.0, MA(2022.6.19)
% Editor : TaeGyu, Yang, MA of Economics, Korea University
%
% This procedure aims to calculate nonlinear panel conditional "ordered" logit estimates.
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
% stat.para : estimated structural parameter vector with MDE
% stat.vcov : variance-covariance matrix
% stat.se : standard error for each parameter
% stat.tv : t-value for each parameter
% stat.pv : p-value
% stat.influ : influence function
% stat.Psi : Psi matrix for MDE 
% stat.overid : overidentification test statistics and p-value 
if nargin==2; print=""; step=0.5; iterlim=500; algorithm="BHHH";
elseif nargin==3; step=0.5; iterlim=500; algorithm="BHHH";
elseif nargin==4; iterlim=500; algorithm="BHHH";
elseif nargin==5; algorithm="BHHH"; end
if isstring(print)~=1; print=string(print); end
if isstring(algorithm)~=1; algorithm=string(algorithm); end
T=size(y,2); R=size(unique(y),1);
if iscell(y)==1; for iter=1:T; y0(:,iter)=y{iter}; end; clearvars y; y=y0; end
if iscell(x)==1; for iter=1:T; x0(:,:,iter)=x{iter}; end; clearvars x; x=x0; end
k=size(x,2); n=size(x,1); 
rf = []; h = [];
for iter = 1:R-1
    for t = 1:T
        ycat(:,:,t) = gendummy(y(:,t));
    end
    z0 = reshape(sum(ycat(:,1:iter,:),2)>0, [n,T]);
    stat0 = clogit(z0, x, print, step, iterlim, algorithm);
    rf = [rf; stat0.para]; h = [h, stat0.influ];
    stat.rf(iter).para=stat0.para;
    stat.rf(iter).yt=z0;
    stat.rf(iter).xt=x;
end
invwn = inv(h'*h); psi = genPsi(R, T, k);
invpwp = inv(psi'*invwn*psi); vcov = invpwp;
sf = invpwp*psi'*invwn*rf;
overid = (rf - psi*sf)'*invwn*(rf - psi*sf);

stat.para=sf;
stat.vcov=vcov;
stat.se=sqrt(diag(stat.vcov));
stat.tv=stat.para./stat.se;
stat.pv=2*tcdf(abs(stat.tv), n*T-size(stat.para,1), 'upper');
stat.influ=h*invwn*psi*(vcov);
stat.Psi=psi;
stat.overid=[overid, chi2cdf(overid, size(rf,1), 'upper')];
if print=="print"; ShowTable(stat, T, R, k, n); end
end

function result=ShowTable(stat, T, R, k, n)
result=[round(stat.para,3), round([stat.se, stat.tv, stat.pv],2)];
hd0=["t"+num2str((2:T)')+" - t"+num2str((1:T-1)')];
hd=[];
for iter=1:R-1
    hd = [hd; hd0 + "- g"+num2str(iter)];
end
hd=[hd;"b"+num2str((1:k)')];
result=[hd, result];
disp("=============================================")
disp("      <Ordered Response Conditional Logit>")
disp(" ")
disp("                                            Wave : "+num2str(T))
disp("                                    Sample Size : "+num2str(n))
disp("---------------------------------------------")
disp("  Parameter    |    Est.  |  Std.Err. |   tv.  |  pv.  ")
disp("---------------------------------------------")
disp(result);
disp("---------------------------------------------")
disp("Over Idendification Test : ")
disp(stat.overid);
disp("=============================================")
end

function psi = genPsi(R, T, k)
Psi0 = kron(eye(R-1),eye(T-1)); psi=[];
for r = 1:R-1
    idx = (1+(T-1)*(r-1) : (T-1)*r);
    psi = [psi; [Psi0(idx,:), zeros(T-1, k); zeros(k, (T-1)*(R-1)), eye(k)]];
end
end