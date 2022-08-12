function stat=MundlakRE(y, x, c, T, print)
% stat=MundlakRE(y, x, c, T, print)
%
% Version 1.0(2022.5.30)
% Editor : TaeGyu, Yang, MA of Economics, Korea University
%
% This procedure aims to calculate linear panel "Correlated Random Effect Estimator" suggested by Mundlak.
%
% Input:
% y : dependent variable
% x : matrix of time-varying variables
% c : matrix of time-constant variables
% T : # of wave
% print : input "print" if wanting displaying result table.
%
% Output :
% stat is a structure object
% stat.para : MDE estimate vector
% stat.vcov : variance-covariance matrix
% stat.se : standard error for each parameter
% stat.tv : (ols-0)/se for each parameter
% stat.pv : p-value for each parameter
% stat.wald : overidentification test statistics and p-value
% stat.Psi : Psi matrix for MDE
% stat.RF : Reduced form parameter
% stat.influ : Influence matrix
n=size(y{1}, 1); kx=size(x{1},2); [nc, kc]=size(c);
if nargin==4; print=""; end
if isstring(print)~=1; print=string(print); end;
if nc==T*n; c=reshape(c, [n, kc*T]); c=c(:,1:kc); end

mx=0; for iter=1:T; mx=mx+x{iter}; end; mx=mx/T;
h=[]; pi=[];
for iter=1:T
    w=[ones(n,1), c, x{iter}-mx, mx];
    invww=inv(w'*w); 
    p(:,iter)=invww*(w'*y{iter});
    v=y{iter}-w*p(:,iter);
    h=[h, (w.*v)*invww]; %Influence Function
    pi=[pi; p(:,iter)]; %Reduced Form Parameter
end
h=h'; Omega=h*h'; invOmega=inv(Omega);
genPsi=@(b)(psi_mundlak(b, T, kc, kx));
Psi=gradp(ones(T+kc+kx+kx,1), genPsi);

vcov=inv(Psi'*invOmega*Psi); MDE=vcov*Psi'*invOmega*pi;
se=sqrt(diag(vcov)); tv=MDE./se; pv=2*tcdf(abs(tv), n-size(w,2), 'upper');
Wald=(pi-Psi*MDE)'*invOmega*(pi-Psi*MDE); Wald_pv=1-chi2cdf(Wald, size(pi,1));

stat.MDE=MDE; stat.vcov=vcov; stat.se=se; stat.tv=tv; stat.pv=pv;
idx=(T+kc+1:1:T+kc+kx)';
stat.b_result=[MDE(idx,:), se(idx,:), tv(idx,:), pv(idx,:)];
stat.wald=[Wald, Wald_pv]; stat.Psi=Psi; stat.RF=pi; stat.influ=h;

if print=="print"
    header=repmat("a", [T,1])+num2str((1:1:T)')+repmat("+m0",[T,1]);
    header=[header; repmat("g", [kc,1])+num2str((1:1:kc)')+repmat("+mc",[kc,1])];    
    header=[header; repmat("b", [kx,1])+num2str((1:1:kx)')];
    header=[header; repmat("mx",[kx,1])+num2str((1:1:kx)')];
    disp("==========================================================")
    disp("                <Chamberlain's Random Effect MDE>")
    disp("                                                    (Mundlak Correction)")
    disp("                                                       i = 1, ... , "+num2str(n));
    disp("                                                       t = 1, ..., "+num2str(T));
    disp("Model : ")
    disp("y(i,t) = a(t) + c(i)'*g + x(i,t)'*b + e(i) + u(i,t)")
    disp("e(i) = m0*1 + c(i)'*mc + x(i,1)'*mx + ... + x(i,T)'*mx + v(i,t)")
    disp("---------------------------------------------------------")
    disp("  Parameter  |  MDE  | Std.Err. | |t-value| | p-value")
    disp("---------------------------------------------------------")
    disp([header, round(MDE,3), round(se,2), round(abs(tv),2), round(pv,2)])
    disp("---------------------------------------------------------")
    disp("Over Identification Test")
    disp("H0 : RF = PSI * SF")
    disp("(Wald, p-value) = " + "( "+num2str(round(Wald,3))+", "+num2str(round(Wald_pv,3))+" )")
    disp("==========================================================")
end
end

function mat=psi_mundlak(b, T, kc, kx)
mat=[];
for iter=1:T
    mat=[mat;
        [b(iter,:);
        b(T+1:T+kc,:);
        b(T+kc+1:T+kc+kx,:);
        b(T+kc+1:T+kc+kx,:) + b(T+kc+kx+1:T+kc+2*kx,:)]
        ];
end
end