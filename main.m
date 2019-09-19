data = wine;
tic;
col= size(data,2);
data = data(:,1:col);%获取数据
label = data(:,col);%获取数据
%data = addnoise( data,0.05);%向数据中增加噪声水平
N_MAX = 100;%实验运行的次数
k = length(unique(label));%数据的类簇数目
d = floor(0.3*col);%降维后的数据
RR=[];
FFM=[];
PP=[];
KK=[];
RRT=[];
for i=1:N_MAX
    [ ~,result,~,~,~] = NCluster(data,d,5*k,k,20);
    [R,FM,P,K,RT] = Evaluation(result,label);
    RR=[RR,R];
    FFM=[FFM,FM];
    PP=[PP,P];
    KK=[KK,K];
    RRT=[RRT,RT];
end
disp(['R的均值为：',num2str(mean(RR)),'$\pm$',num2str(std(RR))]);
disp(['FM的均值为：',num2str(mean(FFM)),'$\pm$',num2str(std(FFM))]);
disp(['P的均值为：',num2str(mean(PP)),'$\pm$',num2str(std(PP))]);
disp(['K的均值为：',num2str(mean(KK)),'$\pm$',num2str(std(KK))]);
disp(['RT的均值为：',num2str(mean(RRT)),'$\pm$',num2str(std(RRT))]);
toc;

function data = addnoise( data,level )%主函数
%data:增加噪声的原数据  level:噪声水平
num = floor(level*size(data,1));
samples = randperm(size(data,1),num);%取样
for i=1:size(samples,2)
    for j=1:size(data,2)-1
        data(i,j)=data(i,j)+10*rand(1,1);
    end
end
end
