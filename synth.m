function stat=synth(Yt,Yc,Zt,Zc,T_pre, print, header)
% stat=synth(Yt,Yc,Zt,Zc,print, header)
%
% Version 1.0 (2021.11.21)
% Editor : TaeGyu Yang, MA of Economics, Korea University
%
% input
% (1) Yt, Yc : Outcomes for treated, and candidates
% (2) Zt, Zc : Covariates-Outcome Block for treated, and candidates
% (3) T_pre : Pre_Period
% (4) print : input "plot" if you want to draw trend plot
%
% output
% stat is structure
% (1) stat.V : Positive Definite Diagonal Matrix
% (2) stat.W : Optimal Weight
% (3) stat.synth : Trend for the synthetic control group(Yc*W)
% (4) stat.diff : Yt(t) - Yc(t,j)*w
% (5) stat.mse_pre : Mean squared prediction error for the pre-treated period
% (6) stat.mse : Mean squared prediction error for the whole period
%
% <Objective Function of Synthetic Control>
% w(v) = Arg.Max. -(Zt - Zc*w)'*V*(Zt-Zc*w)
% v = Arg.Max. -(Yt - Yc*w(v))'*(Yt - Yc*w(v))' for pre_period
%
% Reference
% A.Abadie(2010), 'Synthetic Control Methods for Comparative Case Stutdies', JASA
% M.J. Lee(2021), 'Difference in Differences and Beyound', 93 pp, Parkyoung
if nargin==5; print=" "; header=[];
elseif nargin==6; header=[]; end
if isstring(print)==0; print=string(print); end
if isstring(header)==0; header=string(header); end
if size(header,2)>0; header=header'; end
T_pre=T_pre(end); T=size(Yt,1);
% Step1
v0=size(Zc,1); obj=@(v)( msev(v, Yt, Yc, Zt, Zc).mse );
options = optimset('fmincon'); bL=zeros(size(v0));
v_opt = fmincon(obj,v0,[],[],[],[],bL,[],[],options);

% Step2
H = Zc'*diag(v_opt)*Zc; f = - Zt'*diag(v_opt)*Zc; options = optimset('quadprog');
statw=msev(v_opt,Yt,Yc,Zt,Zc); best_w = abs(statw.w_opt);

stat.W=best_w; stat.V=diag(v_opt); stat.synth=Yc*best_w; stat.diff=Yt-Yc*best_w;
stat.mse_pre=mean(stat.diff(1:T_pre,:).^2);
stat.mse=mean(stat.diff.^2);
if print=="plot"; clc; plotting(stat,Yt,Yc, T_pre, T,header); end
end

function result=plotting(stat,Yt, Yc, T_pre, T, header)
result=[]; figure; 
diff=stat.diff;
m0=min(diff)-0.5*std(diff); m1=max(diff)+0.5*std(diff);
plot((1:1:T)',diff, '-o','linewidth',2); hold on
plot((1:1:T)', diff(T_pre,1)*ones(T,1), ':k', 'linewidth',1.5);
plot( (T_pre)*ones(T,1), linspace(m0, m1, T)' , ':k', 'linewidth', 1.5); hold off
% xlabel("Period", 'fontsize', 12); ylabel("Group Difference", 'fontsize',12);
% title("Outcome Difference between the Treated & the Synthetic Control Group", 'fontsize', 15)
xticks((1:1:T)); if size(header,1)>1; xticklabels(header); end
axis([1 T m0 m1 ]); box on; grid on;

figure
w=stat.W; Ycw=stat.synth;
m0=min(min(Ycw)-0.5*std(Ycw), min(Yt)-0.5*std(Yt));
m1=max(max(Ycw)+0.5*std(Ycw), max(Yt)+0.5*std(Yt));
plot((1:1:T)', Ycw, '-*', 'linewidth', 2); hold on
plot((1:1:T)', Yt, '-o', 'linewidth', 2);
plot((1:1:T)', Yt(T_pre,1)*ones(T,1), ':k', 'linewidth',1.5);
plot( (T_pre)*ones(T,1), linspace(m0, m1, T)' , ':k', 'linewidth', 1.5); hold off
legend("Synthetic Control Group", "Treatment Group", 'fontsize',15);
xticks((1:1:T)); if size(header,1)>1; xticklabels(header); end
axis([1 T m0 m1]); box on; grid on;
end

function stat=msev(v0, Yt, Yc, Zt, Zc)
v=v0; H = Zc'*diag(v)*Zc; f = - Zt'*diag(v)*Zc; l = size(Zc,2);
w_opt=quadprog(H,f,[],[],ones(1,l),1,zeros(l,1),ones(l,1));
w_opt=abs(w_opt); stat.w_opt=w_opt;
mse=sum((Zt-Zc*w_opt).^2); stat.mse=mse;
end