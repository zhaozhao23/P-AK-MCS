clear all;clc;
mu=[1 1 0.1 0.5 1 1];sigma=[0.05 0.1 0.01 0.05 0.2 0.2];
w0=@(x) sqrt((x(:,2)+x(:,3))./x(:,1));
g=@(x) 3*x(:,4)-abs(2*x(:,5)./(x(:,1).*w0(x).^2).*sin(w0(x).*x(:,6)/2));
n_dim=6;%n是维数
N_sample=1e5;%num是抽样数
x(:,1)=normrnd(mu(1),sigma(1),N_sample,1);
x(:,2)=normrnd(mu(2),sigma(2),N_sample,1);
x(:,3)=normrnd(mu(3),sigma(3),N_sample,1);
x(:,4)=normrnd(mu(4),sigma(4),N_sample,1);
x(:,5)=normrnd(mu(5),sigma(5),N_sample,1);
x(:,6)=normrnd(mu(6),sigma(6),N_sample,1);
GX=g(x);
pf=sum(GX<0)/N_sample;