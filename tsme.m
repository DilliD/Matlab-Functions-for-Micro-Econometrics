function stat=tsme(qi1, a0, qi2, b0, step, print, iterlim, algorithm, header)
% stat=tsme(qi1, a0, qi2, b0, step, print, iterlim, algorithm, header)
%
% Version 1.0 (2022.4.2.)
% Editor : Tae Gyu, Yang, MS of Korea University
%
% This procedure offers estimator for Two Stage M-estimator with Nuisance Parameter
% Reference : MJ.Lee(2008), "Micro-Econometrics : Method of Moments and LDV", 102p
%
% Input
% qi1 : Objective function for nusance parameter
% a0 : Initial Value for Maximazing qi1
% qi2 : Objective function for second stage
%       ex) qi2=@(a,b)( -(y - x*b - ...) ), a should be nuisance parameter in the first stage
% b0 : Initial Value for Maximazing qi2
% step : Convergence Speed
% print : input "print" if you want to See result table
% iterlim : Upper limit for numerical iteration
% algorithm : Optimization Algorithm, "BHHH", "NR", "GD", "fminsearch"
% header : Name of Regressors
%
% Output
% stat.para_nsc : Estimated Nuisance Parameter
% stat.para : Estimated Parameter for Second Stage Model
% stat.vcov : Modified Variance Covariance Matrix
% stat.se : Standard Error
% stat.tv : T-value
% stat.pv : P-value corresponding to the t-value
% stat.influ1 : N by K Influence Function for step1
% stat.influ2 : N by K influence function for step2
% stat.link : Link Matrix, see Lee(2008)
% stat.convg : Whether optimization converges or not
if nargin==5; print="print"; iterlim=500; algorithm="BHHH"; header=[];
elseif nargin==6; iterlim=500; algorithm="BHHH"; header=[];
elseif nargin==7; algorithm="BHHH"; header=[];
elseif nargin==8; header=[];
end
if isstring(print)~=1; print=string(print); end
if isstring(algorithm)~=1; algorithm=string(algorithm); end
if size(header,2)>1; header=header'; end
n=size(qi1(a0),1); ka=size(a0,1); kb=size(b0,1);

% Step1 : First Stage
result1=opt(qi1, a0, step, print, iterlim, algorithm);
if result1.converge~=1
    disp("Optimization Algorithm for First Stage did not converge");
    stat.convg=result1.converge;
    return
end
para_nsc=result1.best;
s1=gradp(para_nsc, qi1); invH1=hessp(para_nsc, qi1); invH1=inv(0.5*(invH1+invH1'));
vcov_nsc_inf=inv(s1'*s1);
vcov_nsc_rob=invH1*vcov_nsc_inf*invH1;
influence=(-invH1)*s1';

% Step2 : Second Stage
qi2b=@(b)(qi2(para_nsc, b));
result2=opt(qi2b, b0, step, print, iterlim, algorithm);
stat.convg=result2.converge;
if result2.converge~=1
    disp("Optimization Algorithm for Second Stage did not converge");
end
para=result2.best;
s2=gradp(para, qi2b); invH2=hessp(para, qi2b); invH2=inv(0.5*(invH2+invH2'));

% Step3 : Variance Modification
L=linkmat(para,para_nsc, qi2);
delta=s2'+L*influence;
vcov=invH2*(delta*delta')*invH2;

% Result
stat.para_nsc=para_nsc; stat.vcov_nsc_inf=vcov_nsc_inf; stat.vcov_nsc_rob=vcov_nsc_rob;
stat.para=para; stat.vcov=vcov; stat.se=sqrt(diag(vcov)); stat.tv=para./(stat.se);
stat.pv=2*tcdf(abs(stat.tv), n-kb, 'upper');
stat.influ1=influence';
stat.influ2=delta';
stat.link=L;

if print=="print"
    if (size(header,1)==0)+(size(header,1)>kb)==1
        for iter=1:kb; hd(iter,1)="x"+num2str(iter); end
    else
        hd=header;
    end
    disp("============================================")
    disp("            <Two Stage M-Estimator>")
    disp("--------------------------------------------")
    disp("Regressor |  est.  |  std.err   |   tv   |  pv")
    disp("--------------------------------------------")
    disp([hd, round([stat.para, stat.se, stat.tv, stat.pv],3)])
    disp("--------------------------------------------")
    disp("sample size : "+num2str(n))
    disp("============================================")
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

function L=linkmat(para, para_ncs, qi2, h) % Link Matrix
if nargin<4; h=[0.00001,0.00001]; end
ka=size(para_ncs,1); kb=size(para,1);
eyeka=eye(ka); eyekb=eye(kb); h=0.0001; L=zeros(kb,ka);
for j1=1:kb
     for j2=1:ka
        hj1=h*eyekb(:,j1); hj2=h*eyeka(:,j2);
        upper=sum(qi2(para_ncs+hj2, para+hj1))-sum(qi2(para_ncs+hj2, para-hj1));
        lower=sum(qi2(para_ncs-hj2, para+hj1))-sum(qi2(para_ncs-hj2, para-hj1));
        L(j1,j2)=(upper-lower)./(4*h*h);
    end
end
end