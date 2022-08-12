function stat=heckman(y, d, x, w, step, print, robust, iterlim, algorithm, header)
% stat=heckman(y, d, x, w, step, print, robust, iterlim, algorithm, header)
%
% Version 1.0 (2022.4.2)
% Editor : Tae Gyu, Yang, MS of Korea University
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
% step : Convergence Speed
% print : input "print" if you want to See result table
% iterlim : Upper limit for numerical iteration
% algorithm : Optimization Algorithm, "BHHH", "NR", "GD", "fminsearch"
% header : Name of Regressors
%
% Output
% stat : outcome is structure form
% stat.model0 : Basic Model
% stat.model1~3 : Modified Models for Robustness check
% para_nsc : Estimated Nuisance Parameter
% para : Estimated Parameter for Second Stage Model
% vcov : Modified Variance Covariance Matrix
% se : Standard Error
% tv : T-value
% pv : P-value corresponding to the t-value
% influ : N by K Influence Function Value
% link : Link Matrix, see Lee(2008)
% pRsq : Pseudo R squared for First Stage
if nargin==4; step=0.1; print="print"; robust=""; iterlim=1000; algorithm="BHHH"; header=[];
elseif nargin==5; print="print"; robust=""; iterlim=1000; algorithm="BHHH"; header=[];
elseif nargin==6; robust=""; iterlim=1000; algorithm="BHHH"; header=[];
elseif nargin==7; iterlim=1000; algorithm="BHHH"; header=[];
elseif nargin==8; algorithm="BHHH"; header=[];
elseif nargin==9; header=[];
end
if isstring(print)~=1; print=string(print); end
if isstring(robust)~=1; robust=string(robust); end
if isstring(algorithm)~=1; algorithm=string(algorithm); end
if size(header,2)>1; header=header'; end

n=size(y,1); kx=size(x,2); kw=size(w,2);
a0=inv(w'*w)*w'*d; b0=inv(x'*(d.*x))*x'*(d.*y); %Initial Value

invmill=@(a)( normpdf(w*a)./normcdf(w*a) );
qi1=@(a)( d.*log(normcdf(w*a)) + (1-d).*log(1-normcdf(w*a)) );
syms b [kx+1,1];
qi2=@(a,b)( -d.*( y-[x, invmill(a)]*b ).^2  );

stat0=tsme(qi1, a0, qi2, [b0; 1], step, 0, iterlim, algorithm); convg0=stat0.convg;
if convg0==0
    disp("Optimizatioin does not converges(Model 0)"); stat=[];
elseif convg0==1
    an=stat0.para_nsc;
    stat.nuisance.para=stat0.para_nsc;
    stat.nuisance.vcov_inf=stat0.vcov_nsc_inf;
    stat.nuisance.vcov_rob=stat0.vcov_nsc_rob;
    stat.nuisance.pRsq=cal_pRsq(w, stat.nuisance.para);
    stat0=rmfield(stat0,'para_nsc'); stat0=rmfield(stat0,'vcov_nsc_inf');
    stat0=rmfield(stat0,'vcov_nsc_rob');
    stat.model0=stat0;
    stat.model0.name=["E[y|x,d=1]=xb + r*lamb(w'a)"];
end

if (robust=="robust")*(convg0==1)==1
    syms b [kx+2,1];
    qi2_1=@(a,b)( -d.*( y-[x, (w*a), ((w*a).^2)]*b ).^2 );
    qi2_2=@(a,b)( -d.*( y-[x, invmill(a), (invmill(a).^2)]*b ).^2 );
    syms b [kx+1,1];
    qi2_3=@(a,b)(  -d.*( y-[x, invmill(a), ( 1-(w*a).*invmill(a) )]*b ).^2  );
    
    stat0=tsme(qi1, an, qi2_1, [b0; rand(2,1)], step, 0, iterlim, algorithm); convg1=stat0.convg;
    if stat0.convg==0
        disp("Optimizatioin does not converges(Model 1)"); 
    else
        stat0=rmfield(stat0,'para_nsc');
        stat0=rmfield(stat0,'vcov_nsc_inf');
        stat0=rmfield(stat0,'vcov_nsc_rob');
        stat.model1=stat0;
        stat.model1.name=["E[y|x,d=1]=xb + g1*(wa) + g2*(wa)^2"];
    end
    
    stat0=tsme(qi1, an, qi2_2, [b0; rand(2,1)], step, 0, iterlim, algorithm); convg2=stat0.convg;
    if stat0.convg==0
        disp("Optimizatioin does not converges(Model 2)");
    else
        stat0=rmfield(stat0,'para_nsc');
        stat0=rmfield(stat0,'vcov_nsc_inf');
        stat0=rmfield(stat0,'vcov_nsc_rob');
        stat.model2=stat0;
        stat.model2.name=["E[y|x,d=1]=xb + g1*lamb + g2*lamb^2"];
    end
    stat0=tsme(qi1, an, qi2_3, [b0; rand(2,1)], step, 0, iterlim, algorithm); convg3=stat0.convg;
    if stat0.convg==0
        disp("Optimizatioin does not converges(Model 3)");
    else
        stat0=rmfield(stat0,'para_nsc');
        stat0=rmfield(stat0,'vcov_nsc_inf');
        stat0=rmfield(stat0,'vcov_nsc_rob');
        stat.model3=stat0;
        stat.model3.name=["E[y|x,d=1]=xb + g1*lamb + g2*(1-wa*lamb)"];
    end
end

if (print=="print")*(convg0==1)==1
    if size(header,1)==kx
        hd=[header; "invMill"];
    elseif size(header,1)==kx+1
        hd=header;
    else
        hd=[]; for iter=1:kx; hd=[hd;"x"+num2str(iter)]; end
        hd=[hd; "invMill"]
    end
    disp("=============================================")
    disp("                   <Heckman TSE>")
    disp("-----------------------------------------------")
    disp("Basic Sample Selection Model : "+stat.model0.name);
    disp("Regressor |  Est.  |  Std.Err  |  t.v   |  p.v   ")
    disp("-----------------------------------------------")
    disp([hd, round([stat.model0.para, stat.model0.se, stat.model0.tv, stat.model0.pv],3)]);
    disp("-----------------------------------------------")
    if robust=="robust"
        if convg1==1
            disp("Modified Model 1 : "+stat.model1.name);
            disp("-----------------------------------------------")
            disp([[hd(1:end-1,1);"w'a"; "(w'a)^2"], round(stat.model1.para,3), round([stat.model1.se, stat.model1.tv, stat.model1.pv],2)]);
            disp("-----------------------------------------------")
        end
        if convg2==1
            disp("Modified Model 2 : "+stat.model2.name);
            disp("-----------------------------------------------")
            disp([[hd;"invMill^2"], round(stat.model2.para,3), round([stat.model2.se, stat.model2.tv, stat.model2.pv],2)]);
            disp("-----------------------------------------------")
        end
        if convg3==1
            disp("Modified Model 3 : "+stat.model3.name);
            disp("-----------------------------------------------")
            disp([[hd;"1-(wa)*invMill"], round(stat.model3.para,3), round([stat.model3.se, stat.model3.tv, stat.model3.pv],2)]);
            disp("-----------------------------------------------")
        end
    end
    disp("sample size : "+num2str(n));
    disp("Psuedo R^2 for First Stage : "+num2str(stat.nuisance.pRsq))
    disp("-----------------------------------------------")
end

end

function pRsq=cal_pRsq(w, an)
n=size(w,1); ws=w(:,2:end); dws=(ws-mean(ws)); an_s=an(2:end,:);
pRsq=( an_s'*(1/n)*dws'*dws*an_s )/( 1+ an_s'*(1/n)*dws'*dws*an_s );
end