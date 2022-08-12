function stat=edf(x, print, a)
% stat=edf(x, print, a)
%
% Version 1.0 (2021.10.29)
% Editor : TaeGyu, Yang, MA of Economics, Korea University
%
% This procedure aims to estimate CDF, and its variance for every evaluation point
%
% Input:
% x : One dimensional data, every data point becomes evaluation point
% print : intput "plot" if you want to print out graph
% a : Significance Level
%
% Output :
% stat is a structure object
% stat.x0 : Ordered evaluation point
% stat.Fn : Estimated EDF at every evaluation point
% stat.se : standard error for every evaluation point, sqrt( Fn(1-Fn)/n )
% stat.ci : Confidence interval with 100*(1-a) % level 
if nargin==1; print=" "; a=0.05;
elseif nargin==2; a=0.05; end
if isstring(print)~=1; print=string(print); end
n=size(x,1); Fn=zeros(n,1); x0=sortrows(x);
for i=1:n; ind=x0<=x0(i,:); Fn(i,:)=mean(ind); end
se=sqrt(Fn.*(1-Fn)/n); ci=[Fn-se*abs(icdf('normal', 0.5*a, 0, 1)), Fn+se*abs(icdf('normal', 0.5*a, 0, 1))];
stat.x0=x0; stat.Fn=Fn; stat.se=se; stat.ci=ci;

if print=="print"
    plot(x0, Fn, 'LineWidth', 1.8); hold on
    plot(x0, ci(:,1), ':r', 'LineWidth', 1.2);
    plot(x0, ci(:,2), ':r', 'LineWidth', 1.2);
    plot(x0, zeros(n,1), ':k'); hold off
    axis([min(x0) max(x0) min(ci(:,1)) max(ci(:,2))]);
    xlabel('x','fontsize',15)
    ylabel('F_n (x_0 )','fontsize',15)
    title("Empirical Distribution Function",'fontsize',20)
    legend("EDF",num2str(100*(1-a))+"% CI",'fontsize',15)
    box on; grid on;
end
end
