function stat=trinity(qi, initial, r, c, print)
% stat=trinity(qi, initial, r, c, print)
%
% Version 1.0 (2022.3.29.)
% Editor : Tae Gyu, Yang, MS of Korea University
%
% This procedure aims to calculate Wald, LM, LR test statistics for nonlinear hypothesis
%
% Input
% (1) qi : Individual-wise objective functions, should input maximization form
% (2) initial : Initial Value for numerical optimization
% (3) r : restriction function
% (4) c : target value
% (5) print : input "print" if you want to see result table
%
% Output
% stat : stat is structure
% (1) stat.b_un : Estimated parameter for the unrestricted model
% (2) stat.b_rst : Estimated parameter for the restricted model
% (3) stat.lambda : Estimated lagrangian multiplier for the restricted model
% (4) stat.vcov_rob : Estimated robust variance for the unrestricted model
% (5) stat.wald : Estimated Wald statistics
% (6) stat.waldpv : p-value for the wald
% (7) stat.LM : Estimated LM(Lagrangian Multiplier Test) statistics
% (8) stat.LMpv : p-value for the LM
% (9) stat.Lr : Estimated LR(Likelihood Ratio Test) statistics
% (10) stat.LRpv : p-value for the LR
% (11) stat.df : Degrees of Freedom, rank of the nonlinear restriction
if nargin==3; c=0; print=0;
elseif nargin==4; print=0; end
if isstring(print)~=1; print=string(print); end

Qn=@(a)( -sum(qi(a)) );
nonlinearcon=@(b)( restriction(r, c, b) );
b_un=fminsearch(Qn, initial); score_un=gradp_loc(b_un,qi); invH_un=inv(hessp_loc(b_un, qi));
vcov_un=invH_un*(score_un'*score_un)*invH_un;
R=gradp_loc(b_un,r); df=rank(R);
stat.b_un=b_un;
if (print=="print")+(print=="LM")+(print=="LR")>0
    [b_rst,~,~,~,lambda]=fmincon(Qn, initial, [], [], [], [], [], [], nonlinearcon); score_rst=gradp_loc(b_rst,qi);
    stat.b_rst=b_rst; stat.lambda=lambda.eqnonlin;
end
stat.vcov_rob=vcov_un;

stat.wald=(r(b_un)-c)'*inv(R*vcov_un*R')*(r(b_un)-c); %wald
stat.waldpv=chi2cdf(stat.wald,df,'upper');
if (print=="print")+(print=="LM")>0
    stat.LM=sum(score_rst)*inv(score_rst'*score_rst)*sum(score_rst)'; %LM
    stat.LMpv=chi2cdf(stat.LM,df,'upper');
end
if (print=="print")+(print=="LR")>0
    stat.LR=2*sum( qi(b_un)-qi(b_rst) ); % %LR
    stat.LRpv=chi2cdf(stat.LR, df, 'upper');
end

stat.df=df;
if (print=="print")+(print=="LM")+(print=="LR")+(print=="Wald")>0
    print_table(stat,r,print);
end
end

function result=print_table(stat,r,print)
    result=[];
    disp('==================================')
    disp("  <Non-Linear Hypothesical Test>")
    disp('----------------------------------')
    disp('H0 : ')
    disp(r)
    disp(['degrees of freedom = ', num2str(stat.df)])
    disp('----------------------------------')
    disp('         statstics   p-value')
    disp(['Wald : ', num2str(round(stat.wald,3)),'  ', num2str(round(stat.waldpv,3))])
    if (print=="print")+(print=="LM")>0
        disp(['  LM : ', num2str(round(stat.LM,3)),'  ', num2str(round(stat.LMpv,3))])
    end
    if (print=="print")+(print=="LR")>0
        disp(['  LR  : ', num2str(round(stat.LR,3)),'  ', num2str(round(stat.LRpv,3))])
    end
    disp('==================================')
end

function [constineq, consteq]=restriction(r, c, b0)
constineq=[]; consteq=c-r(b0);
end

function gr=gradp_loc(a,f) %Numerical Gradient
[n,~]=size(f(a)); k=size(a,1); eyek=eye(k); h=0.0001; gr=zeros(n,k);
for j=1:k; hj=h*eyek(:,j); gr(:,j)=(f(a+hj)-f(a-hj))./(2*h); end
end

function hess=hessp_loc(a,qi) %Numerical Hessian
k=size(a,1); eyek=eye(k); h=0.0001; hess=zeros(k,k);
for j1=1:k
     for j2=1:k
        hj1=h*eyek(:,j1); hj2=h*eyek(:,j2);
        upper=sum(qi(a+hj1+hj2))-sum(qi(a+hj1-hj2));
        lower=sum(qi(a-hj1+hj2))-sum(qi(a-hj1-hj2));
        hess(j1,j2)=(upper-lower)./(4*h*h);
    end
end
end
