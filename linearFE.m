function stat = linearFE(y, x, cv, trend_power, print)
% stat = linearFE(y, x, cv, trend_power, print)
% 
% Version 1.0, (2022.7.7)
% Editor : TaeGyu, Yang, MA of Economics, Korea University
%
% This procedure aims to calculate nonlinear panel conditional logit estimates.
% Input :
% y : n by T dependent variable matrix
% x : n by kx by T time-varying regressor tensor
% cv : n by kcv time-invariant regressor with "time-varying parameter", default = [];
% trend_power : order of trend polynomial
% print : input "print" if you want to display result table
%
% Output :
% stat is a structure object
% stat.para : estimated parameter vector
% stat.vcov : variance-covariance matrix
% stat.se : standard error for each parameter
% stat.tv : t-value for each parameter
% stat.pv : p-value
% stat.influ : influence function
if nargin == 2; cv=[]; trend_power = 0; print = "";
elseif nargin == 3; trend_power = 0; print = "";
elseif nargin == 4; print = ""; end
if isstring(print)~=1; print=string(print); end
T = size(y,2); p = trend_power;
if istable(y)==1; for t=1:T; yt(:,t) = y{t}; end; y=[]; y=yt; end
if istable(x)==1; for t=1:T; xt(:,:,t) = x{t}; end; x=[]; x=xt; end
n = size(y,1); kx = size(x,2); kcv = size(cv,2);
if p ==0; Dt = eye(T); Dt=Dt(:,2:end);
elseif p>0; Dt = []; for iter=1:p; Dt=[Dt, ((1:T).^iter)']; end
end
%%%%% Within Group Estimation %%%%%
eyeT=eye(T);
q=eyeT - ones(T,1)*ones(1,T)/T; xqx = 0; xqy = 0;
for i = 1:n
    xt = []; for t = 1:T; xt = [xt; x(i,:,t)]; end
    if kcv==1
        eyeTcv = kron(eyeT(:,2:end),cv(i,1));
        xi{i} = [Dt, xt, eyeTcv];
    elseif kcv==2
        eyeTcv = [kron(eyeT(:,2:end),cv(i,1)), kron(eyeT(:,2:end),cv(i,2))];
        xi{i} = [Dt, xt, eyeTcv];
    elseif kcv==0
        xi{i} = [Dt, xt];
    end
    yi{i} = y(i,:)';
    xqx = xqx + xi{i}'*q*xi{i};
    xqy = xqy + xi{i}'*q*yi{i};
end
invxqx = inv(xqx); WIT = invxqx*xqy; stat.para = WIT; 
xquuqx = 0;
for i = 1:n
    ui{i} = yi{i} - xi{i}*WIT;
    xquuqx = xquuqx + xi{i}'*q*ui{i}*ui{i}'*q*xi{i};
    influ(i, :) = (invxqx*xi{i}'*q*ui{i})';
end
stat.vcov = invxqx*xquuqx*invxqx;
stat.se = sqrt(diag(stat.vcov)); stat.tv = stat.para./stat.se;
stat.pv = 2*tcdf(abs(stat.tv), n-size(stat.para,1), 'upper'); stat.influ=influ;
if print=="print"; ShowTable(stat, p, kx, kcv, n, T); end
end

function result = ShowTable(stat, p, kx, kcv, n, T)
result = [round(stat.para,3), round([stat.se, abs(stat.tv), stat.pv],2)];
if p ==0
    hd0 = "t"+num2str((2:T)')+" - t1";
elseif p>0
    hd0 = "a"+num2str((1:p)');
end
if kcv>0
    hd1=[];
    for iter=1:kcv
        hd1 = [hd1; "h"+num2str(iter)+"("+num2str((2:T)')+")-h"+num2str(iter)+"(1)"];
    end
elseif kcv==0
    hd1 = [];
end
hd = [hd0; "bx"+num2str((1:kx)'); hd1];
disp("==================================================")
disp("               <Fixed Effect Estimation>")
disp("--------------------------------------------------")
disp("* Within Group Estimator")
disp("    para      est.       std.err      |tv|        pv")
disp("--------------------------------------------------")
disp([hd, result]);
disp("--------------------------------------------------")
disp("Sample Size : "+num2str(n));
disp("Wave : "+num2str(T));
if p==0
    disp("Nonparametric Trend");
elseif p==1
    disp("Trend : " +num2str(p)+"st order Polynomial");
elseif p==2
    disp("Trend : " +num2str(p)+"nd order Polynomial");
elseif p==3
    disp("Trend : " +num2str(p)+"rd order Polynomial");
elseif p>3
disp("Trend : " +num2str(p)+"th order Polynomial");
end
disp("==================================================")
end