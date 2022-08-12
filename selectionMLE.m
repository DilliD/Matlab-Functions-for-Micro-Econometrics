function stat=selectionMLE(y, d, x, w, initial, step, print, iterlim, algorithm)
% stat=selectionMLE(y, d, x, w, initial, step, print, iterlim, algorithm)
%
% Version 1.0 (2022.3.22)
% Editor : Tae Gyu, Yang, MA of Korea University
%
% This procedure offers estimator for Heckman's Two Stage Estimation for Sample Selection Model
% d = 1[ w'a + u >0 ]
% y = x'b + v  and (u,v)~Multi-Variate Normal
% Reference : MJ.Lee(2008), "Micro-Econometrics : Method of Moments and LDV"
%
% Input
% y : Outcome
% d : Binary for Participation
% x : Covariates for 2nd Stage Equation
% w : Covariates for 1st Stage Equation
% Initial : Initial Value for numerical optimization
% step : Convergence Speed
% print : input "print" if you want to See result table
% iterlim : Upper limit for numerical iteration
% algorithm : Optimization Algorithm, "BHHH", "NR", "GD", "fminsearch"
%
% Output
% stat : outcome is structure form
% para : Estimated Parameter
% vcov : Modified Variance Covariance Matrix
% se : Standard Error
% tv : T-value
% pv : P-value corresponding to the t-value
if nargin==5; step=0.5; print="print"; iterlim=1000; algorithm="BHHH";
elseif nargin==6; print="print"; iterlim=1000; algorithm="BHHH";
elseif nargin==7; iterlim=1000; algorithm="BHHH";
elseif nargin==8; algorithm="BHHH";
end
if isstring(print)~=1; print=string(print); end
if isstring(algorithm)~=1; algorithm=string(algorithm); end
n=size(y,1); kx=size(x,2); kw=size(w,2); M=1;

rho=@(z)( atan(z)/(0.5*pi) ); %rho : [-inf, inf] -> (-1, 1)
p1=@(a,b,t,s)( normcdf(-w*a) );
p2=@(a,b,t,s)( (1/s)*normpdf( (y-x*b)/s ) );
p3=@(a,b,t,s)( normcdf( ( w*a + (rho(t)/s)*(y-x*b) )/sqrt(1-rho(t)^2) ) );
qi0=@(a,b,t,s)(  (1-d).*log( p1(a,b,t,s) ) + d.*( log( p2(a,b,t,s) ) + log( p3(a,b,t,s) ) ) );
qi=@(c)(  M*qi0( c(1:kw,:),c(kw+1:kw+kx,:), c(end-1,:), c(end,:) ) );

loopiter=0; looplim=10;
while loopiter<looplim
    result=opt(qi,initial,step,print,iterlim,algorithm);
    convg=result.converge;
    if convg==1; break; end
    if convg==0; initial=result.best; end
    if isnan(result.best)==1; initial=(0.5+rand(1,1))*initial0; step=0.8*step0; end
    loopiter=loopiter+1;
end
mle=result.best; maximand=sum(qi(mle)); score=gradp(mle,qi); vcov=score'*score/(M^2);
est=@(a)[ a(1:end-2,:); rho(a(end-1,:)); a(end,:) ];
gr=gradp(mle, est); vcov=gr'*vcov*gr;
mle=est(mle);

if convg==1
    stat.para=mle; stat.max=maximand; stat.vcov=vcov;
    stat.se=sqrt(diag(stat.vcov)); stat.tv=mle./stat.se;
    stat.pv=2*tcdf(abs(stat.tv),n-size(mle,1),'upper');
    clc; disp('< Numerical Optimization Algorithm converges >')

    if print=="print"
        header=[repmat("an", [kw,1])+num2str((1:1:kw)'); repmat("bn", [kx,1])+num2str((1:1:kx)');"rho"; "sigma"];
        result_table(stat,algorithm, n, header);
    end
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
    result=BHHH(qi, initial, step, print, iterlim);
elseif algorithm=="NR"
    result=NR(qi, initial, step, print, iterlim);
elseif algorithm=="GD"
    result=grd_desc(qi, initial, step, print, iterlim);
end
end

function result=result_table(stat, algorithm, n, header)
disp('=================================================')
disp('           <MLE for Sample Selection Model>')
disp(' ')
disp("Model : (u,v)|w ~ N(0, [1, rho; rho, sig]")
disp("           d = 1[ w'a + u >0 ]")
disp("           y = x'b + v")
disp('-------------------------------------------------')
disp('Parameter |  Est.  | Std.Err  |  t-value   |  p-value')
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
disp('=================================================');
result=[];
end