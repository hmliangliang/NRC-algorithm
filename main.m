data = wine;
tic;
col= size(data,2);
data = data(:,1:col);%��ȡ����
label = data(:,col);%��ȡ����
%data = addnoise( data,0.05);%����������������ˮƽ
N_MAX = 100;%ʵ�����еĴ���
k = length(unique(label));%���ݵ������Ŀ
d = floor(0.3*col);%��ά�������
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
disp(['R�ľ�ֵΪ��',num2str(mean(RR)),'$\pm$',num2str(std(RR))]);
disp(['FM�ľ�ֵΪ��',num2str(mean(FFM)),'$\pm$',num2str(std(FFM))]);
disp(['P�ľ�ֵΪ��',num2str(mean(PP)),'$\pm$',num2str(std(PP))]);
disp(['K�ľ�ֵΪ��',num2str(mean(KK)),'$\pm$',num2str(std(KK))]);
disp(['RT�ľ�ֵΪ��',num2str(mean(RRT)),'$\pm$',num2str(std(RRT))]);
toc;

function data = addnoise( data,level )%������
%data:����������ԭ����  level:����ˮƽ
num = floor(level*size(data,1));
samples = randperm(size(data,1),num);%ȡ��
for i=1:size(samples,2)
    for j=1:size(data,2)-1
        data(i,j)=data(i,j)+10*rand(1,1);
    end
end
end
