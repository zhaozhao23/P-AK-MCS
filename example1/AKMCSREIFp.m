clear all;clc;
g1=@(x) 3+0.1*(x(:,1)-x(:,2)).^2-(x(:,1)+x(:,2))/sqrt(2);
g2=@(x) 3+0.1*(x(:,1)-x(:,2)).^2+(x(:,1)+x(:,2))/sqrt(2);
g3=@(x) (x(:,1)-x(:,2))+6/sqrt(2);
g4=@(x) -(x(:,1)-x(:,2))+6/sqrt(2);
g=@(x) min([g1(x),g2(x),g3(x),g4(x)],[],2);
n_dim=2;%n是维数
N_sample=1e5;%num是抽样数 
u=lhsdesign(12,n_dim);
x_doe = u*6-3;
y_doe = g(x_doe);
x=randn(N_sample,n_dim);
iter=1;
p=2;
while (1)
    kriging_model=dacefit(x_doe,y_doe,'regpoly0','corrgauss',10*ones(1,2),1e-1*ones(1,2),20*ones(1,2));
    [mu,s2]=predictor(x,kriging_model);
    pf(iter)=sum(mu<0)/N_sample;
    beta=mu./sqrt(s2);
    REIF=mu.*(1-2*normcdf(beta))+sqrt(s2).*(2-sqrt(2/pi)*exp(-beta.^2/2));
    [a,b]=max(REIF);
    if a<=0;break;end
    x_new=x(b,:);
    for j=1:p-1
        for i=1:N_sample
            dx=norm(x(i,:)-x(b,:));%上一个更新点和所有设计点的距离
            correlation(i,1)=feval(kriging_model.corr,kriging_model.theta,dx);
        end
        REIF=REIF.*(1-correlation);
        [~,b]=max(REIF);
        x_new=[x_new;x(b,:)];
    end
    y_new=g(x_new);
    x_doe=[x_doe;x_new];
    y_doe=[y_doe;y_new];
    iter=iter+1;
end
%Plot
grid_interv = 0.1;
[xs1, xs2] = meshgrid(-6:grid_interv:6);
G_true=g([xs1(:),xs2(:)]);
G_true = reshape(G_true, size(xs1));
[G_k,~]=predictor([xs1(:),xs2(:)],kriging_model);
G_k = reshape(G_k, size(xs1));
figure;
contour(xs1, xs2, G_true,'levellist',0,'LineStyle','-', 'LineWidth', 2.0, 'Color',[0.39, 0.47, 0.64]);hold on;
contour(xs1, xs2, G_k,'levellist',0,'LineStyle','-', 'LineWidth', 1.5, 'Color','k');hold on;
aaa=[7 8];bbb=[0 1];ccc=[0 1];
h1=plot(aaa,bbb,'LineStyle','-', 'LineWidth', 2.0, 'Color',[0.39, 0.47, 0.64]);
h2=plot(aaa,ccc,'LineStyle','-', 'LineWidth', 2.0, 'Color','k');
h3=scatter(x_doe(1:12,1),x_doe(1:12,2),10,'r','LineWidth',1.1);
h4=scatter(x_doe(13:end,1),x_doe(13:end,2),10,'b','LineWidth',1.5);
xlabel('\itX\rm_1','fontsize',11)
ylabel('\itX\rm_2','fontsize',11)
legend([h1,h2,h3,h4],'True function','Kriging prediction','Initial DoE','Added DoE')
set(gca,'FontSize',15);
set(gca,'FontName','Times New Roman');
axis([-6 6 -6 6]); hold off;