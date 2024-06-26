clear all;clc;
tic;
mu=[1 1 0.1 0.5 1 1];sigma=[0.05 0.1 0.01 0.05 0.2 0.2];
w0=@(x) sqrt((x(:,2)+x(:,3))./x(:,1));
g=@(x) 3*x(:,4)-abs(2*x(:,5)./(x(:,1).*w0(x).^2).*sin(w0(x).*x(:,6)/2));
n_dim=6;%n是维数
N_sample=1e5;%num是抽样数
u=lhsdesign(12,n_dim);
x_doe(:,1)=(u(:,1)*6-3)*sigma(1)+mu(1);
x_doe(:,2)=(u(:,2)*6-3)*sigma(2)+mu(2);
x_doe(:,3)=(u(:,3)*6-3)*sigma(3)+mu(3);
x_doe(:,4)=(u(:,4)*6-3)*sigma(4)+mu(4);
x_doe(:,5)=(u(:,5)*6-3)*sigma(5)+mu(5);
x_doe(:,6)=(u(:,6)*6-3)*sigma(6)+mu(6);
y_doe=g(x_doe);
x(:,1)=normrnd(mu(1),sigma(1),N_sample,1);
x(:,2)=normrnd(mu(2),sigma(2),N_sample,1);
x(:,3)=normrnd(mu(3),sigma(3),N_sample,1);
x(:,4)=normrnd(mu(4),sigma(4),N_sample,1);
x(:,5)=normrnd(mu(5),sigma(5),N_sample,1);
x(:,6)=normrnd(mu(6),sigma(6),N_sample,1);
iter=1;
while (1)
    kriging_model=dacefit(x_doe,y_doe,'regpoly0','corrgauss',1*ones(1,6),0.001*ones(1,6),1000*ones(1,6));
    [mu,s2]=predictor(x,kriging_model);
    pf(iter)=sum(mu<0)/N_sample;
    err=2*sqrt(s2);
    EFF=mu.*(2*normcdf(-mu./sqrt(s2))-normcdf((-err-mu)./sqrt(s2))-normcdf((err-mu)./sqrt(s2)))-sqrt(s2).*(2*normpdf(-mu./sqrt(s2))-normpdf((-err-mu)./sqrt(s2))-normpdf((err-mu)./sqrt(s2)))+err.*(normcdf((err-mu)./sqrt(s2))-normcdf((-err-mu)./sqrt(s2)));
    [a,b]=max(EFF);
    if a<0.001;break;end
    x_new=x(b,:);
    y_new=g(x_new);
    x_doe=[x_doe;x_new];
    y_doe=[y_doe;y_new];
    iter=iter+1;
end
toc;