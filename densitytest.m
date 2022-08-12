function stat=densitytest(x, cutoff, print, mf1, mf2, method)
% stat=densitytest(x, cutoff, print, mf1, mf2, method)
%
% Version 1.0 (2021.10.26)
% Editor : TaeGyu, Yang, MA of Economics, Korea University
%
% This procedure aims to test whether there is simple discontinuity for Prob. Density Function.
% See J.McCrary(2007)
% Note : K(z)=max(0, 1-|z|) : Triangular kernel
% Note : True Bias Formula depends on f''(c+), f''(c-), but we do not know them
%
%Input
% x : Univariate Vector, Target Random Variable
% cutoff : Suspected Discontinuous Point
% print : whether print result table or not, input "print"
% mf1 : Bandwidth Smoothing Parameter
% mf2 : Binsize Smoothing Parameter
% method : Estimation Method
%       method = "McCrary", LLN Based Approach
%       method = "KDE", Kernel Density Estimation Based Approach
%       method = "AKDE", Adjusted Kernel Density Estimation Based Approach
%
%Output
% stat is a structural object, saving important results
% stat.logdifference : lim log(fx(c+)) - lim log(fx(c-))
% stat.var : Asymptotic Variace of logdifference estimator
% stat.se : Standard Error
% stat.zv : z statistics
% stat.pv : p-value
% stat.fnL : Estimated pdf, f(c-)
% stat.fnR : Estimated pdf, f(c+)
% stat.bias : Bias Formula Function, input 2nd derivative value "if you have"

if nargin==1; cutoff=0; print=" "; mf1=0; mf2=0.5; method="McCrary";
elseif nargin==2; print=" "; mf1=0; mf2=0.5; method="McCrary";
elseif nargin==3; mf1=0; mf2=0.5; method="McCrary";
elseif nargin==4; mf2=0.5; method="McCrary";
elseif nargin==5; method="McCrary"; end
if isstring(print)==0; print=string(print); end
if isstring(method)==0; method=string(method); end

n=size(x,1); c=cutoff;

if method=="McCrary"
    if mf2<=0; mf2=0.1; end
    if mf1>=min(5*mf2-1, 4); mf1=min(5*mf2-1, 4)-0.1; end
    delta=(-1/5)-(mf1/5); binsize=std(x)*(n^(-mf2)); h=std(x)*(n^delta);
    stat.bandwidth=h; stat.binsize=binsize;
    % Step1 : Creating Grid
    gx=floor((x-c)/binsize).*binsize+binsize/2+c;
    Xj=unique(gx); J=size(Xj,1); Yj=zeros(J,1);
    for iter=1:J; ind=gx==Xj(iter,1); Yj(iter,1)=sum(ind)/(n*binsize); end
    
    % Step2 : Estimating Statistics, Local Linear Model
    ind=(Xj>c); xj0=Xj-c*ones(J,1);
    ker=@(z)( (1+z).*(z<0).*(z>=-1)+(1-z).*(z>=0).*(z<=1) );
    ker0=ker(xj0/h)/h;
    s0r=sum(ker0.*ind); s1r=sum(ker0.*xj0.*ind); s2r=sum(ker0.*(xj0.^2).*ind);
    s0l=sum(ker0.*(1-ind)); s1l=sum(ker0.*xj0.*(1-ind)); s2l=sum(ker0.*(xj0.^2).*(1-ind));
    fnR=sum(ker0.*(s2r-s1r*xj0).*Yj.*ind/(s2r*s0r-s1r*s1r));
    fnL=sum(ker0.*(s2l-s1l*xj0).*Yj.*(1-ind)/(s2l*s0l-s1l*s1l));
    logdifference=log(fnR)-log(fnL); var=(24/5)*( (1/fnR)+(1/fnL) );
    
elseif method=="KDE"
    delta=(-1/5)-(mf1/5); h=std(x)*(n^delta); eps=mf2*h; stat.bandwidth=h; stat.eps=eps;
    % ker=@(z)( (1+z).*(z<0).*(z>=-1)+(1-z).*(z>=0).*(z<=1) );
    % intk2dz=0.6668; intz2kdz=0.1668;
    ker=@(z)[ (2+4*z).*(z<0).*(z>=-0.5)+(2-4*z).*(z>=0).*(z<=0.5) ];
    intk2dz=1.3325; intz2kdz=0.0417;
    
    % Estimating Statistics
    xi0L=(x-(c-eps)*ones(n,1))/h; kL=ker(xi0L); fnL=sum(kL)/(n*h); vfnL=intk2dz*fnL;
    xi0R=(x-(c+eps)*ones(n,1))/h; kR=ker(xi0R); fnR=sum(kR)/(n*h); vfnR=intk2dz*fnR;
    logdifference=log(fnR)-log(fnL);
    var=[1/fnR, -1/fnL]*[vfnR,0;0,vfnL]*[1/fnR, -1/fnL]';
    
elseif method=="AKDE"
    delta=(-1/3)-(mf1/3); h=std(x)*(n^delta); stat.bandwidth=h; xi0=(x-c*ones(n,1))/h;
    ker=@(z)[ 2*(1+z).*(z<0).*(z>=-1)+2*(1-z).*(z>=0).*(z<=1) ];
    k2dzR=1.3489; zkdzR=0.3355; z2kdzR=0.1662;
    k2dzL=1.3233; zkdzL=-0.3326; z2kdzL=0.1665;
    kL=ker(xi0).*(x<c); fnL=sum(kL)/(n*h); vfnL=fnL.*(k2dzL*-fnL);
    kR=ker(xi0).*(x>=c); fnR=sum(kR)/(n*h); vfnR=fnR.*(k2dzR-fnR);
    logdifference=log(fnR)-log(fnL);
    var=[1/fnR, -1/fnL]*[vfnR,0;0,vfnL]*[1/fnR, -1/fnL]';    
end

%Bias Formula
if method=="McCrary"
    bias=@(d2L, d2R)( 0.05*(sqrt(std(x)^5)*(n^(-mf1/2)))*( (-d2R/fnR) - (-d2L/fnL) ) );
elseif method=="KDE"
    bias=@(d2L, d2R)( intz2kdz*0.5*(sqrt(std(x)^5)*(n^(-mf1/2)))*( (d2R/fnR) - (d2L/fnL) ) );
elseif method=="AKDE"
    bias=@(dL, d2L, dR, d2R)( sqrt(std(x)^3)*sqrt(n^(-delta)) * ( (dR*zkdzR+0.5*h*d2R*z2kdzR)/fnR ) - ((dL*zkdzL+0.5*h*d2L*z2kdzL)/fnL) );
end

% Result
se=sqrt(var/(n*h)); zv=(logdifference-0)/se; pv=2*normcdf(abs(zv), 0, 1, "upper");
stat.logdifference=logdifference; stat.var=var; stat.se=se;
stat.zv=zv; stat.pv=pv; stat.fnR=fnR; stat.fnL=fnL; stat.bias=bias;

if or(print=="print",print=="plot")    
    if method=="McCrary"
        resulttable(n,h,stat.logdifference,stat.se,stat.zv,stat.pv,method,binsize);
    elseif method=="KDE"
        resulttable(n,h,stat.logdifference,stat.se,stat.zv,stat.pv,method,eps);
    elseif method=="AKDE"
        resulttable(n,h,stat.logdifference,stat.se,stat.zv,stat.pv,method);
    end
end

if print=="plot"
    if method=="McCrary"; resultplot(c,Xj,Yj); end
end
end

function result=resulttable(n,h,logdifference,se,zv,pv,method,additional)
if nargin==7; additional=[]; end
    disp("===========================================")
    disp('                  <Test Result>')
    disp(' ')
    disp(['                               Sample size : ', num2str(n)])
    if method=="McCrary"
        disp(['                               Binsize        : ', num2str(round(additional,3))])
    elseif method=="KDE"
        disp(['                               epsilon        : ', num2str(round(additional,3))])
    end
    disp(['                               Bandwidth    : ', num2str(round(h,3))])
    disp('-------------------------------------------')
    cat=["Estimator(Dn)    : "; "Standard Error   : "; "Z-Value : "; "P-Value : "];
    disp([cat, round([logdifference; se; zv; pv],4)]);
    disp('-------------------------------------------')
    disp('Note. H0 : There is no Simple Discontinuity')
    disp('Note. sqrt(nh)(Dn-D0) -d-> N(bias, var)')
    if method=="McCrary"
        disp('Note. bias = O(sqrt(nh^5))')
        disp('Note. h -> 0, nh -> inf, nh^5 -> 0, b/h -> 0 as n->inf')
    elseif method=="KDE"
        disp('Note. bias = O(sqrt(nh^5))')
        disp('Note. h -> 0, nh -> inf, nh^5 -> 0 as n->inf')
    elseif method=="AKDE"
        disp('Note. bias = O(sqrt(nh^3))')
        disp('Note. h -> 0, nh -> inf, nh^3 -> 0 as n->inf')
    end
    disp("===========================================")
    result=1;
end

function result=resultplot(c,Xj,Yj)
    scatter(Xj,Yj, 40,'filled'); hold on;
    plot(c*ones(100,1), linspace(min(Yj),max(Yj),100)','LineWidth',1.5); hold off;
    axis([min(Xj) max(Xj) min(Yj) max(Yj)])
%     legend("Discretized Frequency", "Suspected Discontinuity Point", 'fontsize', 16)
%     xlabel("X", 'fontsize', 15); ylabel("dPr(X=x)/dx", 'fontsize', 15);
    box on; grid on;
    result=1;
end