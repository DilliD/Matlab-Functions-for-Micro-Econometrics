function stat = linearPCE(y, x, c, cv, trend_power, factor, print, iterlim)
% stat = linearPCE(y, x, c, cv, trend_power, factor, print, iterlim)
%
% Version 1.0, (2022.5.22)
% Editor : TaeGyu, Yang, MA of Economics, Korea University
%
% This procedure provides Bai(2008)'s Principle Component Estimator.
%
% input
% y : n by T dependent variable matrix
% x : n by kx by T time-varying regressor tensor
% c : n by kc time-invariant regressor matrix
% cv : n by kcv time-invariant regressor with "time-varying parameter", default = [];
% trend_power : order of trend polynomial
% factor : # of macro factors
% print : input "print" if you want to display result table
% iterlim : iteration limit
%
% Output :
% stat is a structure object
% stat.para : estimated parameter vector
% stat.para_bc : estimated parameter vector with bias-correction
% stat.vcov : variance-covariance matrix
% stat.se : standard error for each parameter
% stat.tv : t-value for each parameter
% stat.pv : p-value
% stat.loading : Estimated Loading Matrix(Individual Coefficients for Factors)
% stat.f : Estimated Factor
% stat.minimum : minimum value of objective function
if nargin == 3; cv=[]; trend_power = 2; factor = 1; print = ""; iterlim = 10000;
elseif nargin == 4; trend_power = 2; factor = 1; print = ""; iterlim = 10000;
elseif nargin == 5; factor = 1; print = ""; iterlim = 10000;
elseif nargin == 6; print = ""; iterlim = 10000;
elseif nargin == 7; iterlim = 10000; end
if isstring(print)~=1; print=string(print); end
T = size(y,2); p = trend_power; m = factor;
if istable(y)==1; for t=1:T; yt(:,t) = y{t}; end; y=[]; y=yt; end
if istable(x)==1; for t=1:T; xt(:,:,t) = x{t}; end; x=[]; x=xt; end
n = size(y,1); kx = size(x,2); kc = size(c,2); kcv = size(cv,2);
if size(c,1) > n; c=c(1:n,:); end 
if p + 1 + m > T; m = max([T-p-1,1]); p = T - m - 1; end
Dt = []; for iter=1:p; Dt=[Dt, ((1:T).^iter)']; end
%%%%% Within Group Estimation %%%%%
q=eye(T) - ones(T,1)*ones(1,T)/T;
x0 = [];
for j = 1:kx
    xt(1:n, 1:T)=x(:,j,:);
    x0 = [x0, reshape(xt', [n*T,1])]; 
end
if kcv>0
    eyeTcv=kron(cv, eye(T)); x0 = [x0, eyeTcv(:,2:end)];
end
x0 = [repmat(Dt, [n,1]), x0]; q0 = kron(eye(n), q); y0 = reshape(y', [n*T,1]);
invxqx0 = inv(x0'*q0*x0); xqy0 = x0'*q0*y0; WIT = invxqx0*xqy0;
%%%%% Bai(2008)'s Principle Component Estimation %%%%%
k = p + kx + kc + (T-1)*kcv; b0=[WIT; rand(kc,1)];
if kc>0
    c0 = reshape(repmat(reshape(c, [1, n*kc]), [T,1]), [n*T, kc]);
    x0 = [x0, c0];
end
converge=0;
e00 = reshape( y0 - x0*b0, [T,n]);
bestobj = e00'*e00; bestobj = sum(diag(bestobj));
niter=1;
while niter<=iterlim
    u0 = reshape(y0-x0*b0, [T,n]); uu0 = u0*u0';
    [V, D]=eig(uu0); f=zeros(T,m);
    for j = 1:m; f(:,j)=V(:,T-j+1); end; f =sqrt(T)*f; 
    L = u0'*f/T; Lf = L*f'; Lf = reshape(Lf', [n*T,1]); xx = x0'*x0; xyf = x0'*(y0 - Lf);
    e0 = reshape( y0 - x0*b0 - Lf, [T,n]);
    newobj = sum(diag( e0'*e0 )); 
    b1 = inv(xx)*xyf; % b1 is new estimator
    if print=="print"; disp( num2str([niter, newobj, bestobj]) ); end
    if abs(bestobj - newobj)<0.0001; converge=1; stat.converge = converge; break; end
    if newobj<bestobj; bestobj = newobj; end
    b0 = b1; niter = niter+1;
end
IFE0 = b1; stat.para = IFE0;
e0 = reshape( y0 - x0*IFE0 - Lf, [T,n]);
invLL = inv(L'*L/n); A = L*invLL*L'; QF=eye(T)-f*inv(f'*f)*f';
% Variance
for j = 1:k
    qx = QF*reshape(x0(:,j), [T, n]);
    z(:,j) = reshape((qx - (1/n)*qx*A'),[T*n,1]);
end
zz = z'*z; e=reshape(e0, [n*T,1]); zeez = z'*((e.^2).*z);
nT=n*T; invzz=inv(zz/nT); vcov=invzz*(zeez/nT)*invzz/nT;
% Bias Correction
for iter1 = 1:n
    qxa = 0;
    for iter2=1:n
        idx = (T*(iter2-1)+1:T*iter2)';
        xj = x0(idx,:); xi{iter2} = xj;
        qxa = qxa + QF*xj*A(iter1,iter2);
        vi{iter2}=xj*A(iter1,iter2)/n;
    end
end
ent=e0'; e2=mean(ent.^2); ohm=diag(e2); B1 = 0; B2 = 0;
for iter = 1:n
    xi0 = xi{iter}; vi0 = vi{iter}; Li = L(iter,:)'; sig2i=mean(e0(:,iter).^2)';
    B1 = B1 + (xi0 - vi0)'*(f/T)*invLL*Li*sig2i;
    B2 = B2 + xi0'*QF*ohm*(f/T)*invLL*Li;
end
B1 = -invzz*B1/n; B2 = -invzz*B2/n; bc = IFE0 - (B1/n) - (B2/T);
stat.para_bc = bc; stat.se = sqrt(diag(vcov));
stat.tv = stat.para./sqrt(diag(vcov));
stat.pv = 2*tcdf(abs(stat.tv), T*n - k, 'upper');
stat.vcov = vcov; stat.loading = L; stat.f = f; stat.minimum = newobj;
if print=="print"; ShowTable(stat, p, m, kc, kx, kcv, n, T); end
end

function result = ShowTable(stat, p, m, kc, kx, kcv, n, T)
result2 = [round([stat.para, stat.para_bc],3), round([stat.se, abs(stat.tv), stat.pv],2)];
if kcv>0
    hd1 = ["a"+num2str((1:p)'); "bx"+num2str((1:kx)'); "h("+num2str((2:T)')+")-h(1)"];
    if kc>0
        hd2 = [hd1; "gc"+num2str((1:kc)')];
    elseif kc==0
        hd2 = hd1;
    end
elseif kcv==0
    hd1 = ["a"+num2str((1:p)'); "bx"+num2str((1:kx)')];
    if kc>0
        hd2 = [hd1; "gc"+num2str((1:kc)')];
    elseif kc==0
        hd2 = hd1;
    end
end
disp("==================================================")
disp("                 <Interactive Fixed Effect>")
disp("--------------------------------------------------")
disp("* Bai(2008)'s PCE")
disp("    para      est.   bias correction std.err  |tv|  pv")
disp("--------------------------------------------------")
disp([hd2, result2]);
disp("--------------------------------------------------")
disp("Sample Size : "+num2str(n));
disp("Wave : "+num2str(T));
disp("# of Factors : "+num2str(m));
if p==1
    disp("Trend : " +num2str(p)+"st order Polynomial");
elseif p==2
    disp("Trend : " +num2str(p)+"nd order Polynomial");
elseif p==3
    disp("Trend : " +num2str(p)+"rd order Polynomial");
else
disp("Trend : " +num2str(p)+"th order Polynomial");
end
disp("==================================================")

end