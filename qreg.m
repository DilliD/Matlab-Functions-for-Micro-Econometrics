function stat=qreg(y,x,a,print,step,iterlim,mf,kernel)
% stat=qreg(y,x,a,print,step,iterlim,mf,kernel)
% Version 1.0 (2022.4.13)
% Editor : TaeGyu, Yang, MA of Economics, Korea University
%
% This procedure aims to estimate parameters under quantile regression framework
%
% Input:
% y : Dependent vector
% x : Independent matrix
% a : Quantile
% print : input "print" if you want to display your results
% step : Iteration Speed
% Iterlim : Iteration Limitation
% mf : Adjusting hyper-parameter for bandwidth
% kernel : Kernel choice for estimating f(u|x) at u=0
%        "normal" : k(z)=@(z)( normpdf(z) );
%        "uniform" : k(z)=@(z)( 0.5*(abs(z)<1) );
%        "quadratic" : k(z)=@(z)( 0.75*(1-z.^2).*(abs(z)<1) ); 
%        "biweight"  : k(z)=@(z)( (15/16).*((1-z.^2).^2).*(abs(z)<1) );
%
% Output :
% stat.para : estimated parameter vector
% stat.se : standard error for each parameter
% stat.tv : t-value for each parameter
% stat.pv : p-value
% stat.vcov : variance-covariance matrix
% stat.bandwidth : bandwidth paramter for nonparametric kernel density estimation
% stat.fu : kernel density estimation for f(u|x) at u=0
% stat.influ : influence function
if nargin==2; a=0.5; print=""; step=0.5; iterlim=200; mf=1; kernel="normal";
elseif nargin==3; print=""; step=0.5; iterlim=200; mf=1; kernel="normal";
elseif nargin==4; step=0.5; iterlim=200; mf=1; kernel="normal";
elseif nargin==5; iterlim=200; mf=1; kernel="normal";
elseif nargin==6; mf=1; kernel="normal";
elseif nargin==7; kernel="normal"; end
if isstring(print)~=1; print=string(print); end
if isstring(kernel)~=1; kernel=string(kernel); end
if kernel=="normal"; ker=@(z)(normpdf(z));
elseif kernel=="uniform"; ker=@(z)( 0.5*( abs(z)<1 ) );
elseif kernel=="quadratic"; ker=@(z)( 0.75*(1-z.^2).*(abs(z)<1) ); 
elseif kernel=="biweight"; ker=@(z)( (15/16).*((1-z.^2).^2).*(abs(z)<1) );end
[n,k]=size(x); a=min( abs(a), 1); % Quantile should be bounded
qi=@(b)( -(y-x*b).*(a -(y<x*b)) ); % Objective Function
stat0=BHHH(qi, inv(x'*x)*x'*y, step, print, iterlim); bhat=stat0.best;
%Variance Estimation
uh=y-x*bhat; h=mf*std(uh)*(n^(-1/5)); % Bandwidth
fu0=mean( ker((uh-0)/h)/h ); % Kernel Density Estimation
invfxx=inv(x'*(fu0.*x)); vcov=invfxx*(a*(1-a)*x'*x)*invfxx;
%Result
stat.converge=stat0.converge;
stat.para=bhat; stat.se=sqrt(diag(vcov));
stat.tv=bhat./sqrt(diag(vcov));
stat.pv=2*tcdf(abs(stat.tv), n-k, 'upper');
stat.vcov=vcov; stat.bandwidth=h; stat.fu=fu0;
stat.influ=((a-(y<x*bhat)).*x)*invfxx; %influence Function
stat.max=sum(qi(stat.para));
if print=="print"; ShowTable(stat, a, n, k, x); end
end

function result=ShowTable(stat, a, n, k, x)
result=[round(stat.para,3), round([stat.se, stat.tv, stat.pv],2)];
hd="x"+num2str( (1:k)' );
if size(unique(x(:,1)),1)==1; hd(1,1)="1"; end
disp('====================================================')
if a==0.5
disp('                      <Median Regression>')
else
disp(['                 <',num2str(100*a),'% Quantile Regression>'])
end
disp(['                                           sample size = ',num2str(n)])
disp('----------------------------------------------------')
disp(' Regressor | Est. | Std.Err. |   tv   |    pv  ')
disp('----------------------------------------------------')
disp([hd, result]);
disp('====================================================')    
end