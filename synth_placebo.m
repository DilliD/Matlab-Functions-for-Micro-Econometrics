function stat=synth_placebo(Yt,Yc,Zt,Zc,T_pre, print, discard, header)
% synth_placebo(Yt,Yc,Zt,Zc,T_pre, print, discard, header)
%
% input
% (1) Yt, Yc : Outcomes for treated, and candidates
% (2) Zt, Zc : Covariates-Outcome Block for treated, and candidates
% (3) T_pre : Pre_Period
% (4) print : input "plot" if you want to draw trend plot
%
% output
% stat is structure
% (1) stat.W : Optimal Weight
% (2) stat.diff : Estimated Dynamic Effect after treatment
% (3) stat.mse_pre : Mean squared prediction error for the pre-treated period
% (4) stat.mse : Mean squared prediction error for the whole period
%
% <Objective Function of Synthetic Control>
% w(v) = Arg.Max. -(Zt - Zc*w)'*V*(Zt-Zc*w)
% v = Arg.Max. -(Yt - Yc*w(v))'*(Yt - Yc*w(v))' for pre_period
%
% Reference
% A.Abadie(2010), 'Synthetic Control Methods for Comparative Case Stutdies', JASA
% M.J. Lee(2021), 'Difference in Differences and Beyound', 93 pp, Parkyoung
if nargin==5; print=" "; discard=0; header=[];
elseif nargin==6; discard=0; header=[];
elseif nargin==7; header=[]; end
if isstring(print)==0; print=string(print); end
if isstring(header)==0; header=string(header); end
if size(header,2)>0; header=header'; end

T_pre=T_pre(end); T=size(Yt,1); J=size(Yc,2)+1;
if discard>0
    todel0=sortrows(datasample((1:1:J-1)',discard,'Replace', false)); todel=zeros(J-1,1);
    for iter=1:discard; todel(todel0(iter,1),1)=1; end
    Zc_dis=Zc; Zc_dis(:,todel==1)=[]; Yc_dis=Yc; Yc_dis(:,todel==1)=[];
    Y0=[Yc_dis, Yt]; Z0=[Zc_dis,Zt]; J0=J-1-discard;
elseif discard==0
    Y0=[Yc, Yt]; Z0=[Zc,Zt]; J0=J-1;
end
W_pseudo=zeros(J0,J0); diff_pseudo=zeros(T,J0);
mse_pre_pseudo=zeros(J0,1);
mse_pseudo=zeros(J0,1);
for iter=1:J0
    Yt_pseudo=Y0(:,iter); Zt_pseudo=Z0(:,iter);
    Yc_pseudo=Y0(:,2:end)*(iter==1)+Y0(:,1:end-1)*(iter==J-1)+[Y0(:,1:iter-1), Y0(:,iter+1:end)]*(iter>1)*(iter<J-1);
    Zc_pseudo=Z0(:,2:end)*(iter==1)+Z0(:,1:end-1)*(iter==J-1)+[Z0(:,1:iter-1), Z0(:,iter+1:end)]*(iter>1)*(iter<J-1);
    stat0=synth(Yt_pseudo, Yc_pseudo, Zt_pseudo, Zc_pseudo, T_pre);
    W_pseudo(:,iter)=stat0.W;
    diff_pseudo(:,iter)=Yt_pseudo - stat0.synth;
    mse_pre_pseudo(iter,:)=mean(diff_pseudo(1:T_pre,iter).^2);
    mse_pseudo(iter,:)=mean(diff_pseudo(:,iter).^2);
end
if discard>0
    stat0=synth(Yt,Yc_dis,Zt,Zc_dis,T_pre);
elseif discard==0
    stat0=synth(Yt,Yc,Zt,Zc,T_pre);
end
W_pseudo=[W_pseudo, stat0.W]; stat.W=W_pseudo;
diff_pseudo=[diff_pseudo, stat0.diff]; stat.diff=diff_pseudo;
mse_pre_pseudo=[mse_pre_pseudo; stat0.mse_pre]; stat.mse_pre=mse_pre_pseudo;
mse_pseudo=[mse_pseudo; stat0.mse]; stat.mse=mse_pseudo;
if print=="plot"; plotting(stat, T_pre, T, J0, header); end
end

function result=plotting(stat, T_pre, T, J0, header)
result=[]; figure;
m0=min(min(stat.diff))-0.5*std(min(stat.diff));
m1=max(max(stat.diff))+0.5*std(max(stat.diff));
for iter=1:J0
    plot((1:1:T)',stat.diff(:,iter), '-','color',[0.6 0.6 0.6],'linewidth',0.5); hold on
end
plot((1:1:T)',stat.diff(:,end), '-','color',[0 0.447 0.741],'linewidth',2); hold on
plot((1:1:T)', zeros(T,1), '--k', 'linewidth',1.5);
plot( 0.5*(2*T_pre+1)*ones(T,1), linspace(m0, m1, T)' , '--k', 'linewidth', 1.5); hold off
xticks((1:1:T)); if size(header,1)>1; xticklabels(header); end
axis([1 T m0 m1 ]); box on; grid on;
end