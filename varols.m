function stat=varols(Y0, order, horizon, print, header, nrep, a, longrun)
% stat=varols(Y0, order, horizon, print, header, nrep, a, longrun)
%
% Version 1.0 (2021.10.29)
% Editor : TaeGyu, Yang, MS of Economics, Korea University
%
% This procedure aims to estimate reduced form VAR(p), tries to identify structural form VAR(p)
% under "Recursive Restriction", and suggests "Impulse-Response Analysis" plot.
% 95% Confidence band for Impulse-Response function is calculated by Bootstrap method.
%
% Your structural model is as following : 
% B0*Y(t) = B1*Y(t-1) + .... + Bp*Y(t-p) + e(t) where Y(t) and e(t) are in R^(1 by k)
% Then reduced form is as following : 
% Y(t) = A1*Y(t-1) + .... + Ap*Y(t-p) + u(t) where Y(t) and u(t) are in R^(1 by k)
%  
% Input
% (1) Y0 : Data for several variables, should set T by k matrix
% (2) Order : This input deterimines order of VAR(p) 
% (3) horizon : This input deterimines upper bound of periods for impulse-response analysis
% (4) print : If you input "print", then OLS table for reduced form would be displayed
% (5) header : Input arrays of variable name
% (6) nrep : This input determines the number of replication for bootstrap iteration
% (7) a : Significance level
% (8) longrun : Impose Long Run Restriction
%
% Output
% (1) stat.B0 : (k by k) estimator for B0 in structural form
% (2) stat.Bp : (k by k by p) Tensor, implying structural form OLS estimators
% (3) stat.compA : Companion form constructed by reduced form estimators
% (4) stat.Ap : (k by k by p) Tensor, implying reduced form OLS estimators
% (5) stat.Sigma : Estimated Variance-Covariance matrix s.t. E[uu']
% (6) stat.vcov : Estimated Variace-Covariance matrix for vec(OLS Estimator)

if nargin==2; horizon=15; print=" "; header=[]; nrep=500; a=0.05; longrun="";
elseif nargin==3; print=" "; header=[]; nrep=500; a=0.05; longrun="";
elseif nargin==4; header=[]; nrep=500; a=0.05; longrun="";
elseif nargin==5; nrep=500; a=0.05; longrun="";
elseif nargin==6; a=0.05; longrun="";
elseif nargin==7; logrun=""; end
if isstring(print)==0; print=string(print); end
if isstring(header)==0; header=string(header); end
if isstring(longrun)==0; longrun=string(longrun); end
if size(header,2)>1; header=header'; end

% Reduced Form Estimation
[T,k]=size(Y0); p=order; H=horizon; Z=ones(T-p,1);
for iter=1:p; Z=[Z, Y0(p+1-iter:T-iter,1:end) ]; end
Y=Y0(p+1:T,:); At=inv(Z'*Z)*Z'*Y; rsd=Y-Z*At;

rsd=Y-Z*At; Sigma=rsd'*rsd/(T-p-1-k*p); vcov=kron(inv(Z'*Z),Sigma);
se=sqrt(diag(vcov)); se_v0=se(1:k,:);

if longrun=="longrun"
    Alr=At(2:end,:)'; [K,n]=size(Alr); p=n/K;
    A1lr=eye(K);
    for j=1:p
        A1lr=A1lr-Alr(:,1+(j-1)*K:j*K);
    end
    Flr=inv(A1lr); H1lr=chol(Flr*Sigma*Flr')'; invB0=A1lr*H1lr; %Long-run Restriction
else
    invB0=chol(Sigma)'; % Reculsive Restriction
end

B0=inv(invB0); v0=B0*At(1,:)'

for iter=1:p
    Ap(:,:,iter)=At(k*iter-k+2:k*iter-k+2+k-1,:)'; Bp(:,:,iter)=B0*Ap(:,:,iter); 
    se_Ap(:,iter)=se( k*iter+1:k*iter+(k^2) ,:);
end
Abar=[At(2:end,:)'; eye(k*(p-1)), zeros(k*(p-1),k)];
stat.B0=B0; stat.Bp=Bp; stat.compA=Abar; stat.V=At(1,:)'; stat.se_V=se_v0;
stat.Ap=Ap; stat.se_Ap=se_Ap; stat.Sigma=Sigma; stat.vcov=vcov;

if or(print=="print",print=="plot")
    resulttable(Y0, order, nrep, stat, header);
end
    
if print=="plot"
    bias=bootbias(Y0,rsd,k,p,nrep,T,At);
    At=biascorrection(At, bias, k, p); %Killian(1997)'s Finite Sample Bias Correction
    rsd=Y-Z*At; Sigma=rsd'*rsd/(T-p);
    invB0=chol(Sigma)'; Abar=[At(2:end,:)'; eye(k*(p-1)), zeros(k*(p-1),k)];
    irf=est_irf(At,k,p,H,invB0, Abar);
    irfB=irf_boot(Y0, rsd, H, k, p, nrep, T, At); stat.irfB=irfB;
    figure; irfplot(irfB, irf, H, k, header, nrep);
end
end

function bias=bootbias(Y0, rsd,k,p,nrep,T,At)
bootAt=zeros(size(At));
for biter=1:nrep
    ind=fix(rand(1,1)*(T-p+1))+1; yb=zeros(T,k); yb(1:p,:)=Y0(ind:ind+p-1,:);
    ind=fix(rand(T-p,1)*(T-p))+ones(T-p,1); ub=zeros(T,k); ub(p+1:T,:)=rsd(ind,:);
    for j=p+1:T
        yb(j,:)=At(1,:)+ub(j,:);
        for jj=1:p
            yb(j,:)=yb(j,:) + yb(j-jj,:)*At((jj-1)*k+2:jj*k+1,: );
        end
    end
    Zb=ones(T-p,1);
    for iter=1:p
        Zb=[Zb, yb(p+1-iter:T-iter,1:end) ];
    end
    Yb=yb(p+1:T,:); newAt=inv(Zb'*Zb)*Zb'*Yb; bootAt=bootAt+newAt/nrep;
end
bias=bootAt-At;
end

function Ab=biascorrection(At, bias, k, p)
Abar=[At(2:end,:)'; eye(k*(p-1)), zeros(k*(p-1),k)];
if ~all(abs(eig(Abar))<1); Ab=At;
else; delta=1;
    while delta>=0
        Ab=At-delta*bias; newAbar=[Ab(2:end,:)'; eye(k*(p-1)), zeros(k*(p-1),k)];
        if all(abs(eig(newAbar))<1); break
        else; delta=delta-0.01;end
    end
end
end

function irf=est_irf(At, k, p, H, invB0, Abar) % Estimating Impulse Response
J=[eye(k,k), zeros(k,k*(p-1))]; eyek=eye(k); irf=zeros(H+1,k,k);
for j=1:k
    irf(1,:,j)=(invB0*eyek(:,j))';
    for t=2:H+1
        theta(:,:,t-1)=J*(Abar^(t-1))*J'*invB0; irf(t,:,j)=(theta(:,:,t-1)*eyek(:,j))';
    end
end
end

function irfB=irf_boot(Y0, rsd, H, k, p, nrep, T, At)
J=[eye(k,k), zeros(k,k*(p-1))]; eyek=eye(k);
for biter=1:nrep
    ind=fix(rand(1,1)*(T-p+1))+1; yb=zeros(T,k); yb(1:p,:)=Y0(ind:ind+p-1,:);
    ind=fix(rand(T-p,1)*(T-p))+ones(T-p,1); ub=zeros(T,k); ub(p+1:T,:)=rsd(ind,:);
    for j=p+1:T
        yb(j,:)=At(1,:)+ub(j,:);
        for jj=1:p
            yb(j,:)=yb(j,:) + yb(j-jj,:)*At((jj-1)*k+2:jj*k+1,: );
        end
    end
    Zb=ones(T-p,1);
    for iter=1:p
        Zb=[Zb, yb(p+1-iter:T-iter,1:end) ];
    end
    Yb=yb(p+1:T,:); Atb=inv(Zb'*Zb)*Zb'*Yb; rsdb=Yb-Zb*Atb;
    biasb=bootbias(yb, rsdb, k, p, 1000, T, Atb);
    Atb=biascorrection(Atb, biasb, k, p);
    rsdb=Yb-Zb*Atb;
    invB0b=chol(rsdb'*rsdb/(T-p))';
    Abarb=[Atb(2:end,:)'; eye(k*(p-1)), zeros(k*(p-1),k)];
    irfB(:,:,:,biter)=est_irf(Atb, k, p, H, invB0b, Abarb);
end
end

function result=irfplot(irfB, irf, H, k, header, nrep)
LW=[1.5, 1.2]; result=[];
for iter1=1:k %Response of Y_iter1
    for iter2=1:k %Shock of e_iter2
        subplot(k, k, k*(iter1-1)+iter2)
        pct=prctile(reshape(irfB(:,iter1,iter2,:),[H+1,nrep])', [5, 50 95]);
        plot((0:H)',irf(:,iter1,iter2),'LineWidth',LW(1)); hold on
        plot((0:H)', pct(1,:)', ':k', 'LineWidth', LW(2));
        plot((0:H)', pct(3,:)', ':k', 'LineWidth', LW(2));
        plot(linspace(0,H,100)',zeros(100,1),'LineWidth',1.2); hold off
        if size(header,1)==k; titlename="e_{t,"+header(iter2)+"} -> "+header(iter1)+"_{t}";
        elseif size(header,1)~=k; titlename="e_{t,"+num2str(iter2)+"} -> y_{t,"+num2str(iter1)+"}"; end
        title(titlename,'fontsize',15); xlabel("period",'fontsize',15);box on; grid on; xlim([0 H])
    end
end
legend("IRF", "90% CI", 'fontsize', 15)
end

function result=resulttable(Y0, order, nrep, stat, header)
[T,k]=size(Y0); result=[]; p=order;
cat0=[]; cat1=[];
for i=1:k
    cat0=[cat0; "V_"+num2str(i)];
    for j=1:k
        cat1=[cat1; "A_"+num2str(i)+","+num2str(j)];
    end
end
Ap=stat.Ap; se_Ap=stat.se_Ap; Sigma=stat.Sigma;
disp("==========================================")
disp('  <OLS Estimation for Reduced Form VAR>')
disp(['                                 Sample Size : ',num2str(T)])
disp(['                                 Model Order : ',num2str(order)])
disp(['                                 # of replication : ', num2str(nrep)])
disp('------------------------------------------')
disp('V : ')
disp('  Element      Est.       S.E.')
disp( [cat0, (round(stat.V,3)), (round(stat.se_V,3))] );
for j=1:p
    disp('------------------------------------------')
    disp("A"+num2str(j))
    disp('  Element      Est.       S.E.')
    disp( [cat1, (round(reshape(Ap(:,:,j)',[k^2,1]),3)), (round(se_Ap(:,j),3))] );
end
disp('------------------------------------------')
disp("E[U'U] : ")
disp( round(Sigma,3) );
disp('------------------------------------------')
disp('Note')
if size(header,1)==k
    for iter=1:k; disp("Y_"+num2str(iter)+" : "+header(iter)); end
end
if max(abs(eig(stat.compA)))<1; disp('!! The VAR system is Stable')
elseif sum((abs(eig(stat.compA)) > 1))>=1; disp('!! The VAR system is unstable'); end
disp("==========================================")
end

function v = vec(a)
v = a(:);
end