function stat=ols(y,x,print,robust,header,cluster)
% stat=ols(y,x,print,robust,header,cluster)
%
% Version 1.0 (2021.9.1)
% Editor : TaeGyu, Yang, MA of Economics, Korea University
%
% This procedure aims to calculate "OLS" estimator
%
% Input:
% y : dependent variable
% x : matrix of explanatory variables, with containing ones(n,1)
% print : deciding whether printing result table or not
% robust : binary input deciding whether using robust variance or homoskedastic variance
% header : Variable name
% cluster : cluster category
%
% Output :
% stat is a structure object
% stat.para : estimated parameter vector
% stat.fit : fisher information matrix
% stat.rsd : sandwich form robust variance matrix
% stat.influ : influence function
% stat.sig2 : variance-covariance matrix
% stat.R2 : standard error for each parameter
% stat.adjR2 : t-value for each parameter
% stat.se : standard error for each parameter
% stat.tv : (ols-0)/se for each parameter
% stat.pv : p-value for each parameter

if nargin==2; print=""; robust=""; header=[]; cluster=[];
elseif nargin==3; robust=""; header=[]; cluster=[];
elseif nargin==4; header=[]; cluster=[];
elseif nargin==5; cluster=[]; end
if isstring(robust)~=1; robust=string(robust); end
if isstring(print)~=1; print=string(print); end
% if robust=="cluster"; variance_type="Cluster Variance";
% elseif robust=="robust"; variance_type="Heteroskedastic Robust Variance";
% else; variance_type="Homoskedastic Variance"; end
[n, k]=size(x); invxx=inv(x'*x);
stat.para=invxx*x'*y; %OLS Coefficient Estimator
stat.fit=x*stat.para; %Fitted Value
stat.rsd=y-stat.fit; %Residual
stat.influ=(stat.rsd.*x)*invxx;
rsd2=(stat.rsd).^2;
stat.sig2=stat.rsd'*stat.rsd/(n-k); %Estimated Sigma^2
dyh=stat.fit - mean(stat.fit); dy=y - mean(y);
stat.R2=(dyh'*dyh)/(dy'*dy); %Centered R^2
stat.adjR2=(n/(n-k))*stat.R2;

%Variance-Covariance Matrix
if robust=="cluster"
    variance_type="Cluster Variance";
    xxg=zeros(k,k); xuuxg=zeros(k,k); G=size(cluster,2);
for iter=1:G
    xg=x; xg(cluster(:,iter)==0,:)=[];
    uhg=stat.rsd; uhg(cluster(:,iter)==0,:)=[]; uhg2=uhg.^2;
    xxg=xxg+(xg'*xg); xuuxg=xuuxg+xg'*(uhg2.*xg);    
end
stat.vcov=pinv(xxg)*((G/(G-1))*xuuxg)*pinv(xxg);
elseif robust=="robust"
    variance_type="Heteroskedastic Robust Variance";
    stat.vcov=invxx*(x'*(rsd2.*x))*invxx; %HR Variance
else
    variance_type="Homoskedastic Variance";
    stat.vcov=stat.sig2*invxx; %Homoskedastic Variance
end
stat.se=sqrt(diag(stat.vcov)); %Standard Error
stat.tv=stat.para./stat.se; %t-value
stat.pv=2*tcdf(abs(stat.tv),n-k,'upper'); %p-value

if (print=="print")+(print=="plot")>0
    
    if size(header,1)==k-1; header=["intercept";string(header)];
    elseif size(header,1)==k; header=string(header);
    else; header="intercept"; for i=1:k-1; header=[header;"x"+num2str(i)]; end; end
    disp('================================================')
    disp('              <OLS Estimation Result>')
    disp(['                                      Sample Size = ', num2str(n)])
    disp('------------------------------------------------')
    disp('    Regressor    Coeff.      S.E.        t.v.        p.v.')
    disp([header, round([stat.para, stat.se, stat.tv, stat.pv],3)])
    disp('------------------------------------------------')
    disp(['sigma^2 : ', num2str( round(stat.sig2,3) )])
    disp(['   R^2    : ', num2str( round(stat.R2,3) )])
    disp(['adjusted R^2 : ', num2str( round(stat.adjR2,3) )])
    disp("  VCOV   : "+variance_type)
    disp('================================================')
end
if print=="plot"
    figure
    subplot(2,1,1)
    scatter(x(:,2), y); hold on
    xlabel("x_1", 'fontsize', 12); ylabel("y", 'fontsize',12);
    x1=linspace(min(x(:,2)),max(x(:,2))); W=mean(x(:,3:k));
    yline=stat.para(1)+stat.para(2)*x1+W*stat.para(3:k);
    plot(x1,yline,'LineWidth',1.5);
    text(0.5*(min(x1)+max(x1))+std(x1),0.5*(min(y)+max(y)),'Y=Xb+u','Fontsize',10)
    axis([min(x1)-0.5*std(x1),max(x1)+0.5*std(x1) min(y)-0.5*std(y) max(y)+0.5*std(y)])
    xlabel("x_1", 'fontsize', 12); ylabel("Regression Line", 'fontsize',12);
    grid on; box on;
    
    subplot(2,1,2)
    plot([stat.rsd,zeros(n,1)])
    ylabel('Residuals');
    title('OLS Residuals');
    grid on; box on;
end

end

function dummy=gendummy(x)
% dummy=gendummy(x)
% This procedure generates indicators for every category of input x
[n,k]=size(x);
if k==1 
    ux=unique(x);
    for iter=1:size(ux,1)
        dummy(1:n,iter)=(x==ux(iter,1));
    end
else
    disp("input column vector"); dummy=nan; end
end