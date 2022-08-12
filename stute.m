function stat=stute(model, x, y, print, replication, step)
% stat=stute(model, x, y, print, replication, step)
%
% Version 1.0 (2021.9.30)
% Editor : TaeGyu, Yang, MA of Economics, Korea University
%
% This Procedure aims to test whether the model is correctly specified or not
% based on nonparametric Stute Test : See W.Stute(1997)
%
% Input
% model : n by 1 vector for E[y|x]
%           For example, if your model is E[y|x]=Pr[y=1|x]=normcdf(xa)
%           Define model=@(a,x0)[ normcdf(x0*a) ];
% x and y : x is a n by k matrix of all regressors, including ones(n,1)
%             y is a n by 1 vector of dependent variable
% print : whether display result table or not, input either 'print' or "print"
% replication : # of bootstrap replication, default value is 300
% step : related to convergence speed of BHHH algorithm, default value is 0.1
%
% Output
% stat is a structure object
% stat.stute : stute statistics
% stat.pv : p-value of stute statistics calculated by bootstrap method
% stat.sampledist : bootstrap distribution of stute statistics from pseudo sample

% Basic Setup
if nargin==3; print=" "; step=0.1; replication=300;
elseif nargin==4; step=0.1; replication=300;
elseif nargin==5; step=0.1; end
if isstring(print)~=1; print=string(print); end
[n,k]=size(x); one=ones(n,1); initial=inv(x'*x)*x'*y;
obj=@(a)[ -(y-model(a,x)).^2 ]; 

% Estimating Stute-Statistics
[est,~]=m_est(obj, initial, step, 0); mempdf=zeros(n,1); rsd=y-model(est,x);
for iter=1:n
    onexx=kron( one, x(iter,:) ); ind=(x>onexx)==0; ind=( sum(ind) ==k );
    mempdf(iter,1)=sum( sum(ind)'.* rsd )/sqrt(n);
end
stat.stute=mempdf'*mempdf/n;

% Estimating Pseudo Distribution through Wild-Bootstrap
stutedist=zeros(replication,1);
for biter=1:replication
    disp("bootstrap iteration = "+num2str(biter));
    vb=(2*(rand(n,1)>0.5)-1); yb=model(est,x)+vb.*rsd;
    newobj=@(a)[ -(yb-model(a,x)).^2 ];  [estb,~]=m_est(newobj, initial, step, " ", 1000, "robust", "BHHH" );
    if isnan(estb)==1
        stutedist(biter,1)=nan;
    else
        rsdB=yb-model(estb,x); mempdf=zeros(n,1);
        for iter=1:n
            onexx=kron( one, x(iter,:) ); ind=(x>onexx)==0; ind=( sum(ind) ==k );
            mempdf(iter,1)=sum( sum(ind)'.* rsdB )/sqrt(n);
        end
        stutedist(biter,1)=mempdf'*mempdf/n;
    end
end
todel=isnan(stutedist); stutedist(todel==1,:)=[]; %delete non-converged result
stat.pv=mean(stutedist>stat.stute); stat.sampledist=stutedist;
% Result
if print=="print"
    clc
    disp('======================================')
    disp('    <Stute Model Specification Test>')
    disp('--------------------------------------')
    disp(" H0 : Following E[y|X] is true specification ")
    disp(" E[y|X] : ")
    disp(model);
    disp('--------------------------------------')
    disp([["Stute Statistics : "; "P-Value : "],round([stat.stute;stat.pv],4)]);
    disp("    # of Replication : "+num2str(replication-sum(todel)))
    disp('======================================')
    
    figure
    histogram(stat.sampledist);
    title("Sample Distribution of Stute Statistics","fontsize",15)
end
end