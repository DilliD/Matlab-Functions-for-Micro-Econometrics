function stat=dyprobit(y, c, x, lag, print, step, ns, initial, iterlim, varying, algorithm, header)
% stat=dyprobit(y, c, x, lag, print, step, ns, initial, iterlim, varying, algorithm, header)
%
% Version 1.0, (2022.5.9)
% Editor : TaeGyu, Yang, MA of Economics, Korea University
% This procedure aims to estimate Random Effect Panel Probit Model
%
% Input:
% y : Binary Dependent Variable, n by T vector
% c : Time-Invariant Regressors, n by kc matrix
% x : Time-Variant Regressors, n by kx by T tensor
% print : deciding whether printing result table or not
% step : hyper-parameter affecting on convergence speed of BHHH algorithm
% ns : # of Simulation, the hyper-parameter is related to the Monte-Carlo Integration
% initial : Initial value for optimization, input string if you want to specify nothing
% iterlim : upper limit of iternation
% Varying : Allowing Time Varying Parameter, default setting is time-invariant parameters
% algorithm : Choose which algorithm would be used for numerical optimization
%     if algorithm="fminsearch" : Use Matlab fminsearch
%     if algorithm="BHHH" : Berndt-Hall-Hall-Hausman Algorithm
%     if algorithm="NR"    : Newton-Raphson Algorithm
%     if algorithm="GD"    : Gradient Descent Algorithm
% header : Variable Name
%
% Output :
% If BHHH algorithm does not converge, then NaN value is assigned to every output 
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
if nargin==3; lag=0; print=""; step=0.5; ns=100; initial="default"; iterlim=500; varyring=""; algorithm="BHHH"; header=[];
elseif nargin==4; print=""; step=0.5; ns=100; initial="default"; iterlim=500; varying=""; algorithm="BHHH"; header=[];
elseif nargin==5; step=0.5; ns=100; initial="default"; iterlim=500; varying=""; algorithm="BHHH"; header=[];
elseif nargin==6; ns=100; initial="default"; iterlim=500; varying=""; algorithm="BHHH"; header=[];
elseif nargin==7; initial="default"; iterlim=500; varying=""; algorithm="BHHH"; header=[];
elseif nargin==8; iterlim=500; varying=""; algorithm="BHHH"; header=[];
elseif nargin==9; varying=""; algorithm="BHHH"; header=[];
elseif nargin==10; algorithm="BHHH"; header=[];
elseif nargin==11; header=[]; end
if isstring(print)~=1; print=string(print); end
if isstring(varying)~=1; varying=string(varying); end
if isstring(algorithm)~=1; algorithm=string(algorithm); end
if isstring(header)~=1; header=string(header); end
T=size(y,2);
if iscell(y)==1; for iter=1:T; y0(:,iter)=y{iter}; end; clearvars y; y=y0; end
if iscell(x)==1; for iter=1:T; x0(:,:,iter)=x{iter}; end; clearvars x; x=x0; end
[nt, kc]=size(c); kx=size(x,2); n=nt*T; v=randn(nt, ns);
if isstring(initial)==1
    if varying=="varying"; g0=(0.5+0.5*rand(1,1))*ones(T+lag+kc+T*kx+1+lag,1);
    elseif varying~="varying"; g0=(0.5+0.5*rand(1,1))*ones(T+lag+kc+kx+1+lag,1); end
else
    g0=initial;
end
if varying=="varying"; qi=@(para)( like_varying(y,c,x,lag,v,para) );
else; qi=@(para)( like_novarying(y,c,x,lag,v,para) );
end
[~,stat]=m_est(qi, g0, step, print, iterlim,"",algorithm, header);
end

function q=like_novarying(y, c, x, lag, v, para)
[nt,T]=size(y); kc=size(c,2); kx=size(x,2); ns=size(v,2);
t0=para(1:T,:); one_ns=ones(1,ns);
if lag==0
    b0=para(T+1:T+kc+kx,:);
    s0=para(T+kc+kx+1,:);
    wb1=t0(1,1)+[c, x(:,:,1)]*b0;
    f=normcdf( (wb1.*one_ns + s0*v).*(2*y(:,1)-1) );
    for iter=2:T
        wbt=t0(iter,1)+[c, x(:,:,iter)]*b0;
        f=f.*normcdf( (wbt.*one_ns + s0*v).*(2*y(:,iter)-1) );
    end    
elseif lag>0
    a0=para(T+1:T+lag,:);
    b0=para(T+lag+1:T+lag+kc+kx,:);
    s0=para(T+lag+kc+kx+1:end,:);
    f=1;
    for iter=1:T
        wbt=t0(iter,1)+[c, x(:,:,iter)]*b0;
        if iter==1
            wb = wbt.*one_ns + s0(iter,1)*v;
        elseif (iter>1).*(iter<=lag)==1
            wbt = wbt+y(:,1:iter)*a0(1:iter,:);
            wb = wbt.*one_ns + s0(iter,1)*v;
        elseif iter>lag
            wbt=wbt+y(:,1:lag)*a0;
            wb = wbt.*one_ns + s0(end,1)*v;
        end
        f=f.*normcdf( wb.*(2*y(:,iter)-1) );
    end
end
q=log( eps + mean(f')' );
end

function q=like_varying(y, c, x, lag, v, para)
[nt,T]=size(y); kc=size(c,2); kx=size(x,2); ns=size(v,2);
t0=para(1:T,:); one_ns=ones(1,ns);
if lag==0
    bc0=para(T+1:T+kc,:);
    bx0=reshape(para(T+kc+1:T+kc+T*kx,:),[kx,T]);
    s0=para(T+kc+T*kx+1,:);
    wb1=t0(1,1)+c*bc0 + x(:,:,1)*bx0(:,1);
    f=normcdf( (wb1.*one_ns + s0*v).*(2*y(:,1)-1) );
    for iter=2:T
        wbt=t0(iter,1)+c*bc0 + x(:,:,iter)*bx0(:,iter);
        f=f.*normcdf( (wbt.*one_ns + s0*v).*(2*y(:,iter)-1) );
    end    
elseif lag>0
    a0=para(T+1:T+lag,:);
    bc0=para(T+lag+1:T+lag+kc,:);
    bx0=reshape(para(T+lag+kc+1:T+lag+kc+T*kx,:),[kx,T]);
    s0=para(T+lag+kc+T*kx+1:end,:);
    f=1;
    for iter=1:T
        wbt=t0(iter,1)+c*bc0 + x(:,:,iter)*bx0(:,iter);
        if iter==1
            wb = wbt.*one_ns + s0(iter,1)*v;
        elseif (iter>1).*(iter<=lag)==1
            wbt = wbt+y(:,1:iter)*a0(1:iter,:);
            wb = wbt.*one_ns + s0(iter,1)*v;
        elseif iter>lag
            wbt=wbt+y(:,1:lag)*a0;
            wb = wbt.*one_ns + s0(end,1)*v;
        end
        f=f.*normcdf( wb.*(2*y(:,iter)-1) );
    end
end
q=log( eps + mean(f')' );
end