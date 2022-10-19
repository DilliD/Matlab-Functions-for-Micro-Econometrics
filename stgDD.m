function stat=stgDD(yt,A,var,print)
% stat = stgDD(yt, A, x, var, print)
% This procedure provide DID estimator under staggered adoption design suggested by Athey & Inbens (2022)
%
% input
% yt : n by T outcomes
% A : n by 1 adoption period
% var : variance option
%       if var = "consv"(default), Athey & Imbens' conservative variance is suggested
%       if var = "LZ", Liang-Zerger cluster variance is suggested
%       if var = "all", both conservative variance and LZ variance are provided :(v_did, v_LZ)
% print : input "print" if you want to display estimation results(not printing is default)
%       if print = "print", estimation table is printed
%       if print = "latex", table command for LaTeX is printed
%
% outcome
% stat.para : estimated DID estimator
% stat.var : estimated variance estimator
% stat.se : estimated standard error
% stat.tv : estimated t-value
% stat.pv : p-value for given t-value
% stat.para : estimated OLS estimator
if nargin==2; var="consv"; print="";
elseif nargin==3; print=""; end
[n,T]=size(yt);
for t=1:T+1
    if t<T+1; wit(:,t)=(A<=t); end
    if sum(A==t)==0
        pa(t,1)=0;
    else
        pa(t,1)=mean(A==t);
    end
    na(t,1)=n*pa(t,1);
    apa(t,1)=t*pa(t,1);
end
dAi=gendummy(A);
if size(unique(A),1)<T+1
    if max(A)==T+1
        dAi=[zeros(n,1), dAi];
    elseif max(A)~=T+1
        dAi=[dAi, zeros(n,1)];
    end
end
mat_pa=sum(pa'.*dAi,2);

% Calculation gamma(Weight on adoption and period)
g=@(t,a)( ((a<=t)-sum(pa(1:t,:))) + (1/T)*(a.*(a<=T)-sum(apa(1:T,:))) + ((T+1)/T)*( (a==T+1) - pa(end,1) ) );
den=0; for t=1:T; den=den+sum(pa((1:T+1)',1).*g(t,(1:T+1)').*g(t,(1:T+1)')); end
gamma=@(t,a)( pa(a,1).*g(t,a)/den );

% Getting Treatment Estimator
for t=1:T
    for a=1:T+1
        if na(a,1)>0
            bYta(a,t)=sum( (A==a).*yt(:,t) )/sum(A==a);
        elseif na(a,1)==0
            bYta(a,t)=0;
        end
    end
end

for a=1:T+1
    for t=1:T
        gbYa(a,t) = gamma(t,a)*bYta(a,t);
    end
end
bYa=sum(gbYa,2); stat.para=sum(sum(gbYa,2),1);

% Variance Estimator
if var=="consv" % Conservative Variance
    for a=1:T+1
        yia(:,a)=yt*gamma((1:T),a); % Y_{i}(a) is estimated by sum_{t}gamma(t,a)Y_{i,t}
        dy=mean(yia(:,a)) - bYa(a,1);
        sa2(a,1)=sum((dy.^2).*(A==a));
    end
    stat.var=sum(sa2./(na.*(na-1)));
elseif var=="LZ" % LZ Cluster variance
    y=reshape(yt,[n*T,1]);
    wi=reshape(wit, [n*T,1]);
    Di=eye(n); Di=repmat(Di(:,1:end-1),[T,1]);
    Dt=kron(eye(T), ones(n,1)); Dt=Dt(:,1:end-1);
    x=[wi, Di, Dt, ones(T*n,1)];
    invxx=inv(x'*x); u=y-x*invxx*x'*y; xuux=0;
    for i=1:n
        idx=n*((1:T)'-1)+i;
        xi=x(idx,:);
        ui=u(idx,:);
        xuux = xuux + xi'*((ui.^2).*xi);
    end
    v_lz0=invxx*xuux*invxx;
    stat.var=v_lz0(1,1);
elseif var=="all"
    for a=1:T
        weight=gamma((a:T)',a); weight=weight./sum(weight);
        yia(:,a)=((A==a).*yt(:,a:T))*weight;
    end
    weight=gamma((1:T)', T+1); weight=weight./sum(weight);
    yia(:,T+1)=((A==T+1).*yt(:,1:T))*weight;

    for a=1:T+1
        dy=mean(yia(:,a)) - bYa(a,1);
        sa2(a,1)=sum((dy.^2).*(A==a));
    end
    stat.var(1,1)=sum(sa2./(na.*(na-1)));

    y=reshape(yt,[n*T,1]);
    wi=reshape(wit, [n*T,1]);
    Di=eye(n); Di=repmat(Di(:,1:end-1),[T,1]);
    Dt=kron(eye(T), ones(n,1)); Dt=Dt(:,1:end-1);
    x=[wi, Di, Dt, ones(T*n,1)];
    invxx=inv(x'*x); u=y-x*invxx*x'*y; xuux=0;
    for i=1:n
        idx=n*((1:T)'-1)+i;
        xi=x(idx,:);
        ui=u(idx,:);
        xuux = xuux + xi'*((ui.^2).*xi);
    end
    v_lz0=invxx*xuux*invxx;
    stat.var(1,2)=v_lz0(1,1);
end
df=(T*n)-(n-1+T+1);
stat.se=sqrt(stat.var);
stat.tv=stat.para./stat.se;
stat.pv=2*tcdf(abs(stat.tv),df,'upper');

if print=="print"
    disp("=============================================")
    disp( "   <Treatment Effect under Staggered DID>")
    disp("                   (see Athey & Imbens (2022), JOE)")
    disp("---------------------------------------------")
    disp("  Effect     Std.Err     t-value      p-value")
    disp(num2str(round([stat.para, stat.se, stat.tv, stat.pv],3)));
    disp("---------------------------------------------")
    disp("Sample Size = "+num2str(n))
    disp("Total Period = "+num2str(T))
    for iter=1:T
        disp("pi("+num2str(iter)+") : "+num2str(pa(iter,:)))
    end
    disp("pi("+num2str(inf)+") : "+num2str(pa(end,1)) )
    disp("---------------------------------------------")
    disp("Note")
    if var=="consv"
        disp("* Variance = Athey's Conservative Variance")
    elseif var=="LZ"
        disp("* Variance = Liang-Zerger Cluster Variance")
    end
    disp("* t ~ student("+num2str(df)+")")
    disp("=============================================")
elseif print=="latex"
    disp("<Est and std.err option>")
    disp("Treatment Effect&"+num2str(round(stat.para,3))+"("+num2str(round(stat.se,3))+")\\");
    disp(" ")
    disp("<Est and |t-value| option>")
    disp("Treatment Effect&"+num2str(round(stat.para,3))+"("+num2str(round(abs(stat.tv),3))+")\\");
end
end

function y=gendummy(x)
ux=unique(x);
for iter=1:size(ux,1)
    y(:,iter)=(x==ux(iter,1));
end
end