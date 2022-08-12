function [mle,stat]=m_est(qi,initial,step,print,iterlim,robust,algorithm, header)
% [mle,stat]=m_est(qi, initial, step, print, iterlim, robust, algorithm, header)
%
% Version 1.0 (2021.10.13)
% Editor : TaeGyu, Yang, MA of Economics, Korea University
%
% This procedure aims to calculate M-estimator based on BHHH algorithm
% Before apply the procedure, define likelihood "Contribution" for your own "MAXIMIZATION" problem
%
% Example defining Objective function : Probit MLE
% qi=@(a)[ y.*log(normcdf(X*a)) + (1-y).*log(1-normcdf(X*a)) ]
%
% Input:
% qi : Likelihood contribution function (Sum(qi)=Likelihood Function)
% initial : initial point of parameter input
% step : hyper-parameter affecting on convergence speed of BHHH algorithm
% print : deciding whether printing result table or not
%     if print="print"; print iteration process and estimation result table
%     if print="plot"; print iteration process, estimation result table, and 3D plot of objective function
% iterlim : upper limit of iternation
% robust : binary input deciding whether using robust variance or fisher information
% algorithm : Choose which algorithm would be used for numerical optimization
%     if algorithm="fminsearch" : Use Matlab fminsearch
%     if algorithm="BHHH" : Berndt-Hall-Hall-Hausman Algorithm
%     if algorithm="NR"    : Newton-Raphson Algorithm
%     if algorithm="GD"    : Gradient Descent Algorithm
% header : Variable Name
%
% Output :
% If BHHH algorithm does not converge, then NaN value is assigned to every output 
% mle : estimated parameter vector 
% stat is a structure object
% stat.para : estimated parameter vector
% stat.information : fisher information matrix
% stat.vcov : variance-covariance matrix
% stat.se : standard error for each parameter
% stat.tv : t-value for each parameter
% stat.pv : p-value
% stat.max : Maximund of likelihood function
% stat.score : Estimated score matrix
% stat.influ : influence function
if nargin==3; iterlim=500; print=0; robust=" "; algorithm='BHHH'; header=[];
elseif nargin==4; iterlim=500; robust=" "; algorithm='BHHH'; header=[];
elseif nargin==5; robust=" "; algorithm='BHHH'; header=[];
elseif nargin==6; algorithm='BHHH'; header=[];
elseif nargin==7; header=[]; end
if isstring(print)~=1; print=string(print); end
if isstring(robust)~=1; robust=string(robust); end
if isstring(algorithm)~=1; algorithm=string(algorithm); end
if size(header,2)>1; header=header'; end
n=size(qi(initial),1); k=size(initial,1);

loopiter=0; looplim=10;
while loopiter<looplim
    result=opt(qi,initial,step,print,iterlim,algorithm);
    convg=result.converge;
    if convg==1; break; end
    if convg==0; initial=result.best; end
    if isnan(result.best)==1; initial=(0.5+rand(1,1))*initial0; step=0.8*step0; end
    loopiter=loopiter+1;
end
mle=result.best;
stat.convg=convg;

if convg==1
    stat.para=mle; 
    stat.max=sum(qi(mle)); 
    stat.score=gradp(mle,qi);
    stat.information=stat.score'*stat.score;
    if robust=="robust"
        hess0=hessp(mle,qi); invhess0=inv(hess0);
        stat.influ=(stat.score)*(-invhess0);
        stat.vcov=invhess0*(stat.information)*invhess0;
    elseif robust~="robust"
        stat.influ=(stat.score)*inv(stat.information);
        stat.vcov=inv(stat.information);
    end
    stat.se=sqrt(diag(stat.vcov)); stat.tv=mle./stat.se;
    stat.pv=2*tcdf(abs(stat.tv),n-k,'upper'); stat.rsq=0;
    clc; disp('< Numerical Optimization Algorithm converges >')

    if (print=="print")+(print=="plot")==1
        if (size(header,1)==0)+(size(header,1)>k)==1
            for iter=1:k; hd(iter,1)="x"+num2str(iter); end
        else
            hd=header;
        end
        result_table(stat,robust,algorithm,n, hd);
    end
    if print=="plot"; ploting(stat,qi); end
elseif convg==0
    clc; disp('!! Numerical optimization algorithm does not converge');
    disp('!! Input other initial value or step-size !!'); mle=nan; stat=nan;
end
end

function result=opt(qi, initial, step, print, iterlim, algorithm)
if algorithm=="fminsearch"
    Qn=@(a)( -sum(qi(a)) ); [mle,~,convg]=fminsearch(Qn,initial);
    result.best=mle; result.max=-Qn(mle); result.converge=convg;
elseif algorithm=="BHHH"
    result=BHHH(qi, initial, step, print, iterlim, 0.000001);
%     result=BHHH(qi, initial, step, print, iterlim);
elseif algorithm=="NR"
    result=NR(qi, initial, step, print, iterlim);
elseif algorithm=="GD"
    result=grd_desc(qi, initial, step, print, iterlim);
end
end

function result=ploting(stat, qi)
lim1=50; lim2=50; Qn=zeros(lim1,lim2);
m1=linspace(0.5*stat.para(2,1), 1.5*stat.para(2,1), lim1)';
m2=linspace(0.5*stat.para(3,1), 1.5*stat.para(3,1), lim2)';
for iter1=1:lim1
    for iter2=1:lim2
        m0=[stat.para(1,1);m1(iter1,1);m2(iter2,1);stat.para(4:end,:)]; Qn(iter1,iter2)=sum(qi(m0));
    end
end
figure; mesh(m1,m2,Qn); box on; grid on;
xlabel("parameter for x1", 'fontsize', 15); ylabel("parameter for x2", 'fontsize', 15);
zlabel("Qn", 'fontsize', 15); title("Objective Function near optimum", 'fontsize', 15);
result=[];
end

function result=result_table(stat, robust, algorithm, n, header)
disp('=================================================')
disp('                       <M-Estimator>')
disp('-------------------------------------------------')
disp('Regressor |  Est.  | Std.Err  |  t-value   |  p-value')
disp('-------------------------------------------------')
disp([header,round([stat.para,stat.se,stat.tv,stat.pv],3)])
disp('-------------------------------------------------')
disp(['sample size = ',num2str(n)])
disp(['maximand = ', num2str(stat.max)])
disp('-------------------------------------------------')
if algorithm=="NR"
    disp("Optimization Algorithm : Newton-Raphson")
elseif algorithm=="GD"
    disp("Optimization Algorithm : Gradient Descent")
else
    disp("Optimization Algorithm : Berndt-Hall-Hall-Hausman")
end
if robust=="robust"; disp('Variance Matrix: invH0 * Info0 * invH0')
elseif robust~="robust"; disp('Variance Matrix : inv Info0')
end
disp('=================================================');
result=[];
end