function stat=LocalPoly(y,x,print,delta,mf,header,order,kernel)
% stat=LocalPoly(y,x,print,delta,mf,header,order,kernel)
%
% Version 1.0 (2021.11.25)
% Editor : TaeGyu, Yang, MA of Economics, Korea University
%
% This procedure aims to calculate "Local Polynomial" estimator
%
% Input:
% y : dependent variable
% x : matrix of explanatory variables, with containing ones(n,1)
% print : if print="print", result table is displayed
% delta : 1st adjusting factor for smoothing parameter, default value is 0
% mf : 2nd adjusting factor for smoothing parameter, default value is 1
% header : Variable name
% order : Order of Local Polynomial Regression, default value is one
% kernel : Choice of kernel, default value is "normal"
% note that smoothing parameter h = mf*std(x)*(n^(-(1+delta)/5));
%
% Output :
% stat is a structure object
% stat.para : estimated parameter vector(local lienar coefficients)
% stat.vcov : variance-covariance matrix for each parameter
% stat.Ey : estimated E[y|x]
% stat.VarEy : estimated variance for E[y|x]
% stat.seEy : estimated standard error for E[y|x]
% stat.Dy : estimated slope of E[y|x]
% stat.varDy : estimated variance for slope of E[y|x]
% stat.seDy : estimated standard error for slope of E[y|x]
% stat.varY : E[y^2|x]- {E[y|x]}^2
% stat.fn : kernel density for random variable y|x
% stat.bandwidth : smoothing parameter
load LocalPoly_MCintegral_Values.mat;
if nargin==2; print=" "; delta=0; mf=1; header=[]; order=1; kernel="normal";
elseif nargin==3; delta=0; mf=1;  header=[];order=1; kernel="normal";
elseif nargin==4; mf=1; header=[]; order=1; kernel="normal";
elseif nargin==5; header=[]; order=1; kernel="normal";
elseif nargin==6; order=1; kernel="normal";
elseif nargin==7; kernel="normal"; end
if isstring(print)==0; print=string(print); end
if isstring(kernel)==0; kernel=string(kernel); end
if isstring(header)==0; header=string(header); end
if size(header,2)>1; header=header'; end
if kernel=="triangle"
    ker=@(z)( (1+z).*(z<0).*(z>=-1)+(1-z).*(z>=0).*(z<=1) );
    theta=trikernel.THETA; invtheta=inv(theta(1:order,1:order));
    eta=trikernel.ETA; eta=eta(1:order,1:order);
    vcov0=invtheta*eta*invtheta;
else
    ker=@(z)( normpdf(z) );
    theta=normkernel.THETA; invtheta=inv(theta(1:order,1:order));
    eta=normkernel.ETA; eta=eta(1:order,1:order);
    vcov0=invtheta*eta*invtheta;
end

[n,k]=size(x); h=mf*std(x)*(n^(-(1+delta)/5));
for iter=1:n
    if (iter>1)*(iter<n)==1 %Leave One Out
        xi=[x(1:iter-1,:);x(iter+1:n,:)]; Y=[y(1:iter-1,:);y(iter+1:end,:)];
    else
        xi=x(2:end,:)*(iter==1)+x(1:end-1,:)*(iter==n); Y=y(2:end,:)*(iter==1)+y(1:end-1,:)*(iter==n);
    end
    xi0=xi-x(iter,:).*ones(n-1,1); 
    k0=ones(n-1,1); hh=1;
    for iter2=1:k; k0=k0.*( ker(xi0(:,iter2)/h(:,iter2))./h(:,iter2) ); hh=hh*h(:,iter2); end
    X=genpoly(xi0,order); W=diag( k0 ); invxwx=inv(X'*W*X);
    bhat=invxwx*(X'*W*Y); rsd=Y-X*bhat;
    vcov=(n*hh)*invxwx*(X'*W*(rsd'*rsd/n)*W*X)*invxwx;
    diagvcov(iter,:)=diag(vcov)';
    bn(iter,:)=bhat';
    
    Y2=Y.^2; bn2=(invxwx*(X'*W*Y2))'; Ey2(iter,:)=bn2(1,1);
end
Ey=bn(:,1);
varY=abs(Ey2-(Ey.^2));
jtfn=ones(n,1); for iter=1:k; jtfn=jtfn.*kerden(x(:,iter),delta," ").fn; end; fn=abs(jtfn);
varYpfn=varY./fn; varEy=varYpfn.*vcov0(1,1);

for iter1=1:order
    Dy(:,:,iter1)=bn(:, k*(iter1-1)+1+1 : k*iter1+1);
    varDy(:,:,iter1)=diagvcov(:,k*(iter1-1)+1+1 : k*iter1+1);
%     varDy(:,:,iter1)=repmat(varYpfn*vcov0(iter1,iter1),1,k); %Variance 계산 수정해야함
end

stat.para=bn;
stat.Ey=Ey; stat.varEy=varEy; stat.seEy=sqrt(varEy/(n*hh));
stat.Dy=Dy; stat.varDy=varDy; stat.seDy=sqrt(varDy/(n*hh));
stat.vcov=vcov0; stat.varY=varY; stat.fn=fn; stat.bandwidth=h;

if print=="plot"; plot_Ey(y,x,stat,k,header); plot_ME(x,stat,k,header); end
if print=="Ey"; plot_Ey(y,x,stat,k,header); end
if print=="Dy"; plot_ME(x,stat,k,header); end
end

function matrix=genpoly(x,order)
n=size(x,1); matrix=ones(n,1);
for p=1:order; matrix=[matrix, x.^p]; end
end

function result=plot_Ey(y,x,stat,k,header)
if size(header,1)==0; k0=k;
elseif size(header,1)>0; k0=size(header,1); end
Ey=stat.Ey; se=stat.seEy;
for iter=1:k0
    figure
    ww=[];
    ww=sortrows([x(:,iter), Ey, Ey-1.96*se, Ey+1.96*se],1);
    scatter(x(:,1), y, 40, [0.75 0.75 0.75],'filled'); hold on
    plot(ww(:,1), ww(:,2), '-', 'color', 0.4*rand(1,3), 'linewidth', 1.8);
    plot(ww(:,1), ww(:,3), ':', 'color', 0.45*ones(1,3), 'linewidth', 1.6);
    plot(ww(:,1), ww(:,4), ':', 'color', 0.45*ones(1,3), 'linewidth', 1.6); hold off
    axis([min(ww(:,1)) max(ww(:,1)) min(ww(:,3))-0.5*std(ww(:,2)) max(ww(:,4))+0.5*std(ww(:,2))]);
    if size(header,1)>0
        xlabel(header(iter), 'fontsize',12); ylabel("E[y|x]", 'fontsize',12);
    else
        xlabel("x"+num2str(iter), 'fontsize',12); ylabel("Marginal Effect", 'fontsize',12);
    end
    legend("Scatter Plot","E[y|x]", "95% CI",'fontsize',15)
    box on; grid on;
end
result=[];
end

function result=plot_ME(x,stat,k,header)
if size(header,1)==0; k0=k;
elseif size(header,1)>0; k0=size(header,1); end
for iter=1:k0
    dydx=stat.Dy(:,iter,1); se=stat.seDy(:,iter,1);
    figure
    ww=sortrows([x(:,iter), dydx, dydx-1.96*se, dydx+1.96*se]);
    plot(ww(:,1), ww(:,2), '-', 'color', 0.4*rand(1,3), 'linewidth', 1.6); hold on
    plot(ww(:,1), ww(:,3), ':', 'color', [0.7, 0.7, 0.7], 'linewidth', 1.4);
    plot(ww(:,1), ww(:,4), ':', 'color', [0.7, 0.7, 0.7], 'linewidth', 1.4);
    m0=min(ww(:,1)); m1=max(ww(:,1));
    plot(linspace(m0,m1,100)', zeros(100,1), ':k', 'linewidth', 1.4); hold off
    axis([min(ww(:,1)) max(ww(:,1)) min(ww(:,3))-0.5*std(ww(:,2)) max(ww(:,4))+0.5*std(ww(:,2))]);
    if size(header,1)>0
        xlabel(header(iter), 'fontsize',12); ylabel("Marginal Effect", 'fontsize',12);
        title("dE[y|X]/d("+string(header(iter))+")",'fontsize',15);
    else
        xlabel("x"+num2str(iter), 'fontsize',12); ylabel("Marginal Effect", 'fontsize',12);
        title("dE[y|X]/dx_"+num2str(iter),'fontsize',15);
    end
    legend("Marginal Effect", "95% CI",'fontsize',15)
    box on; grid on;
end
result=[];
end