function stat=kerden(x, mf, print, a, kernel)
% stat=kerden(x, mf, print, a, kernel)
%
% Version 1.0 (2021.10.29)
% Editor : TaeGyu, Yang, MA of Economics, Korea University
%
% This procedure aims to estimate probability density function, variance, and confidence interval
% with Kernel Estimation Method, for every evaluation point
%
% Input:
% x : One dimensional data, every data point becomes evaluation point
% mf : Smoothing Parameter, h = n^(-0.2 - mf/5)
% print : intput "plot" if you want to print out graph
% a : Significance Level
%
% Output :
% stat is a structure object
% stat.x0 : Evaluation point
% stat.fn : Estimated EDF at every evaluation point
% stat.se : standard error for every evaluation point, sqrt( Fn(1-Fn)/n )
% stat.ci : Confidence interval with 100*(1-a) % level
% stat.bandwidth : smoothing parameter h = n^(-0.2 - mf/5)

if nargin==1; mf=0; print=""; a=0.05; kernel="normal";
elseif nargin==2; print=""; a=0.05; kernel="normal";
elseif nargin==3; a=0.05; kernel="normal";
elseif nargin==4; kernel="normal"; end
if isstring(print)~=1; print=string(print); end
if isstring(kernel)~=1; kernel=string(kernel); end
[n,k]=size(x); h=std(x)*(n^(-0.2-(mf/5)));

if kernel=="normal"
    k0=@(x)( normpdf(x) );
end

integral_K2_dz=1/(2*sqrt(pi)); %Monte Carlo Integration

fn=zeros(n,1); se=zeros(n,1);
for i=1:n
    ker=ones(n-1,1); hh=1;
        if and(i==1, i==n); xi=x(2:n,:)*(i==1)+x(1:n-1,:)*(i==n);
        elseif 1<i<n; xi=[x(1:i-1,:);x(i+1:n,:)]; end %Leave One Out Estimator
    for j=1:k
        hk=h(1,j);  hh=hh*hk;  ker=ker.*k0( ( xi(:,j)-x(i,j)*ones(n-1,1) )./hk ); %Multivariate - Kernel 
    end
    fn(i,1)=mean(ker)/hh; se(i,1)=sqrt(fn(i,1)*integral_K2_dz/(n*hh));
end
ci=[fn-se*abs(icdf('normal',0.5*a,0,1)), fn+se*abs(icdf('normal',0.5*a,0,1))];
ww=sortrows([x(:,1),fn,ci,se]);

stat.x0=ww(:,1); stat.fn=ww(:,2); stat.ci=ww(:,3:4);
stat.se=ww(:,5); stat.bandwidth=h;

if print=="plot"
    lw=1.5;
    plot(ww(:,1),ww(:,2),'LineWidth',lw);
    hold on;
    plot(ww(:,1),ww(:,3),':r','LineWidth',1.2);
    plot(ww(:,1),ww(:,4),':r','LineWidth',1.2); hold off;
    axis([min(ww(:,1)) max(ww(:,1)) min(ww(:,3)) max(ww(:,4))]);
    xlabel('x','fontsize',15)
    ylabel('f_n (x_0 )','fontsize',15)
    title("Kernel Density Estimation",'fontsize',20)
    legend("KDE : f_n ",num2str(100*(1-a))+"% CI",'fontsize',15)
    box on; grid on;
end
end