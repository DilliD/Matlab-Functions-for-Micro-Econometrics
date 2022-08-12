function stat=AllPeriodRE(y, x, c, T, print)
% stat=AllPeriodRE(y, x, c, T, print)
%
% Version 1.0(2022.4.2)
% Editor : TaeGyu, Yang, MA of Economics, Korea University
%
% This procedure aims to calculate linear panel "Correlated Random Effect Estimator" suggested by Chamberlain.
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
w=[ones(n,1), c]; for iter=1:T; w=[w, x{iter}]; end
invww=inv(w'*w); h=[]; pi=[];
for iter=1:T
    p(:,iter)=invww*(w'*y{iter});
    v=y{iter}-w*p(:,iter);
    h=[h, (w.*v)*invww]; %Influence Function
    pi=[pi; p(:,iter)]; %Reduced Form Parameter
end
h=h'; Omega=h*h'; invOmega=inv(Omega);
Psi=construct_Psi(kc, kx, T);

vcov=inv(Psi'*invOmega*Psi); MDE=vcov*Psi'*invOmega*pi;
se=sqrt(diag(vcov)); tv=MDE./se; pv=2*tcdf(abs(tv), n-size(w,2), 'upper');
Wald=(pi-Psi*MDE)'*invOmega*(pi-Psi*MDE); Wald_pv=1-chi2cdf(Wald, size(pi,1));

stat.para=MDE; stat.vcov=vcov; stat.se=se; stat.tv=tv; stat.pv=pv;
idx=(T+kc+1:1:T+kc+kx)'; stat.b_result=[MDE(idx,:), se(idx,:), tv(idx,:), pv(idx,:)];
stat.wald=[Wald, Wald_pv]; stat.Psi=Psi; stat.RF=pi; stat.influ=h;

if print=="print"
    header=repmat("a", [T,1])+num2str((1:1:T)')+repmat("+m0",[T,1]);
    header=[header; repmat("g", [kc,1])+num2str((1:1:kc)')+repmat("+mc",[kc,1])];    
    header=[header; repmat("b", [kx,1])+num2str((1:1:kx)')];
    for iter=1:T
        header=[header; repmat("m",[kx,1])+num2str((1:1:kx)')+repmat("("+num2str(iter)+")",[kx,1])];
    end
    disp("==========================================================")
    disp("                <Chamberlain's Random Effect MDE>")
    disp("                                                       i = 1, ... , "+num2str(n));
    disp("                                                       t = 1, ..., "+num2str(T));
    disp("Model : ")
    disp("y(i,t) = a(t) + c(i)'*g + x(i,t)'*b + e(i) + u(i,t)")
    disp("e(i) = m0*1 + c(i)'*mc + x(i,1)'*m(1) + ... + x(i,T)'*m(T) + v(i,t)")
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

function Psi=construct_Psi(kc, kx, T)
blockeyeT=eye(kx*T); eyeT=eye(T); Psi0=[];
%Part1 : Constant Term Part
for iter=1:T
    Psi0=[Psi0;[ eyeT(iter,:); [zeros(kc+(T+1-1)*kx, T)] ]];
end
%Part2 : Time-Invariant Regressor Part
Psi1=[zeros(1, kc); eye(kc); zeros(T*kx,kc)];
Psi1=repmat(Psi1, [T,1]);
%Part3 : Time-Variant Regressor Part
Psi2=[];
for iter=1:T
    idx=( 1+kx*(iter-1) :1: kx*iter )';
    Psi2=[Psi2; [zeros(1+kc, kx); blockeyeT(:, idx)]];
end
%Part4 : All Regressor Part
Psi3=[zeros(1+kc,T*kx); blockeyeT];
Psi3=repmat(Psi3, [T,1]);
%Part5 : Merging
Psi=[Psi0, Psi1, Psi2, Psi3];
end