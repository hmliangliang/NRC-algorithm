function [ final_data,final_target,MSE_value,center,mse] = NCluster( data,d,k,k_cluster,N_max)%main function
%UNTITLED Summary of this function goes here
%dataΪ����,ÿһ�д���һ��ʵ��
%k��d��Ҫ����lle�㷨��,k��ʾKNN�е�kֵ,dΪlle�㷨��ά�����ݵ�ά��
%radius���lle�㷨������İ뾶,��Ҫ��������������
%k_cluesterΪ��������ص���Ŀ
%��������ֲڼ�ģ���������ݽ��о���
%radius2Ϊ���������,����ֲڼ�����İ뾶
%N_maxΪ�����㷨�����Ĵ���
%�����������ʾ�ĺ������£�
%final_target:�������ľ�������Ӧ��ʵ���ǩ,ÿһ�д���һ��ʵ��,
%MSE_value:ÿһ�ξ������ľ������
%cluster_result:��¼ÿһ�ξ���Ľ��,cluster_result(x).result(i,j)=1��ʾ��x�ξ�����,��j��ʵ�����鵽��i�������
%centerΪ�������ĵ�
col=size(data,2);
%data=data(:,1:(col-1));%����
%����������������������
Atrr_c=[];%����������
Atrr_n=[];%����������
lamba=0.9;%�������ĵ��½��Ƽ���ռ�ı���
N=20;%�������������������ͻ��Ƿ���������ֵ����Ŀ
for i=1:col
   if length(unique(data(:,i)))<=N%˵��������Ϊ������
      Atrr_c=[Atrr_c,data(:,i)];
   else%˵��������Ϊ��ֵ��
      Atrr_n=[Atrr_n,data(:,i)];
   end
end
if isempty(Atrr_n)==1%û����ֵ������
    Atrr_n=[Atrr_n,rand(size(data,1),2)];
end
%d=round(0.3*size(Atrr_n,2));%ȷ������lle�㷨��dֵ
%disp(['����lle�㷨��ά���ά�ȵ���ĿdΪ:d=',num2str(d)]);
Atrr_c=round(Atrr_c);
Atrr_c=Norm_attr(Atrr_c,N);
%�Է��������Խ�������ѡ��,Atrr_cf����ѡ������������
Atrr_cf=FeatureSelection(Atrr_c);
%disp(['�㷨��ά���ά�ȵ���ĿΪ:',num2str(d+size(Atrr_cf,2))]);
%����ֵ�����ݽ��й�һ��
Atrr_n=zscore(Atrr_n);%���б�׼��
eu=randi(size(data,1),1,11);
d1=[];
d2=[];
for i=2:11
    d1=[d1,norm( Atrr_n(eu(1),:)- Atrr_n(eu(i),:))];
    d2=[d2,sum(Atrr_c(eu(1),:)~=Atrr_c(eu(i),:))];
end
radius=mean(d1);
radius2=round(mean(d2))+radius;
%disp(['��ֵ��������ֵradius=',num2str(radius)]);
%disp(['������+��ֵ��������ֵradius2=',num2str(radius2)]);
clear d1;
clear d2;
%[y,Ps]=mapminmax(Atrr_n',0,1);%y�����һ����Ľ��
%�����������ݽ��н�ά
y=Atrr_n';
mydata_n=Neighborhoodlle(y',d,k,radius);
data=[mydata_n,Atrr_cf];
n=size(mydata_n,2);%ǰn������Ϊ����������
final_data=data;
%����ÿ�����ݶ�������򼯺�
neighbor=zeros(size(data,1),size(data,1));%�����������
for i=1:size(data,1)
    for j=i:size(data,1)
        d_distace=n/(size(data,2))*norm(data(i,1:n)-data(j,1:n),2)+(size(data,2)-n)/(size(data,2))*sum((data(i,1+n:size(data,2))~=data(j,n+1:size(data,2))));%��������֮��ľ���
        if d_distace<=radius2%��j��Ԫ�����ڵ�i��Ԫ�ص�����
            neighbor(i,j)=1;
        end 
    end
end
%�����ݽ��о���
p=data(randi(size(data,1),1,k_cluster),:);%���ѡȡ���ӵ㣬ÿһ�д���һ�����ӵ�
%cluster_result=struct('result',zeros(k_cluster,size(data,1)));%��¼ÿһ�ξ���Ľ��,cluster_result(x).result(i,j)=1��ʾ��x�ξ�����,��j��ʵ�����鵽��i�������
MSE_value=[];%����ÿһ�ξ������ľ������
cluster_result=zeros(k_cluster,size(data,1));%��¼���յľ�����
for i=1:N_max
   %1.�������ĵ�����򼯺�
   p1=p;
   result=zeros(k_cluster,size(data,1));%��¼ÿһ���ӵ������,result(i,j)=1��ʾ��j��ʵ�����ڵ�i�����ӵ������
   for j=1:k_cluster
       for x=1:size(data,1)
           d=n/(size(data,2))*norm(p1(j,1:n)-data(x,1:n),2)+(size(data,2)-n)/(size(data,2))*sum((p1(j,1+n:size(data,2))~=data(x,n+1:size(data,2))));%��������֮��ľ���
           if d<=radius2 %����������
               result(j,x)=1;
           end
       end
   end
   %�����Ͻ��Ƽ����½��Ƽ�
   low_approximate=cell(1,k_cluster);
   upper_approximate=cell(1,k_cluster);
   for j=1:k_cluster
       for x=1:size(data,1)
           if all(ismember(find(neighbor(x,:)==1),find(result(j,:)==1)))==1%���������Ӽ�,�����Ͻ��Ƽ�
               low_approximate{j}=[low_approximate{j},x];
           elseif isempty(intersect(find(neighbor(x,:))==1,find(result(j,:)==1)))==0%�����½��Ƽ�
               upper_approximate{j}=[upper_approximate{j},x];
           end
       end
       upper_approximate{j}=[upper_approximate{j},low_approximate{j}];
       %�����ظ���Ԫ��
       upper_approximate{j}=unique(upper_approximate{j});
       low_approximate{j}=unique(low_approximate{j});
       %�������ĵ�
       p1(j,:)=lamba*mean(data(low_approximate{j},:))+(1-lamba)*mean(data(upper_approximate{j},:));
       p1(j,:)=[p1(j,1:n),round(p1(j,n+1:size(data,2)))];%������յľ�������
   end
   %�ж��Ƿ�ﵽ��ֹ����
   %��������ָ��
   mse=0;%�������ĵ�ı仯����
   for j=1:k_cluster
       rr=n/(size(data,2))*norm(p(j,1:n)-p1(j,1:n),2)+(size(data,2)-n)/(size(data,2))*sum(p(j,n+1:size(data,2))~=p1(j,1+n:size(data,2)));
       mse=mse+rr;
       MSE_value=[MSE_value,mse];
   end
   if mse>radius/5%���ĵ�仯�ϴ�,���۹���δ��ɣ���Ҫ�������ĵ�
       p=p1;
       clear p1;
   else%����������,
       break;
   end
end
%ȷ��ÿһ�����ݵĹ��������
for c=1:size(data,1)
    rt=zeros(1,k_cluster);
    for j=1:k_cluster
        rt(j)=Distance(p(j,:),data(c,:),data,n);
    end
    [t,y]=min(rt);%��ȡ���յľ�����,tΪ��С�ľ���ֵ,yΪ���
    cluster_result(y,c)=1;%ȷ�����ݵĹ���
end
%�Ѿ�����ת����ԭ�������ǩ
final_result=cluster_result;%��ȡ���һ�εľ�������ÿһ���д���һ�����
%����������
mse=0;
for j=1:size(final_result,1)
     label=find(final_result(j,:)>0);
     for x=1:length(label)
         tempdata=data(label(x),:);
         mse=mse+norm(tempdata(1:n)-p(j,1:n),2)+sum(tempdata(n+1:size(data,2))~=p(j,1+n:size(data,2)));
     end
end
mse=1/size(data,1)*mse;

final_target=cell(1,k_cluster);%�������ľ�������ÿһ�д���һ��ʵ��
for j=1:k_cluster
%     disp(['j=',num2str(j)]);
%     find(final_result(j,:)==1)
    if isempty(find(final_result(j,:)==1))==0
       final_target{1,j}=find(final_result(j,:)==1);%ת����ԭ����Ӧ�����ǩ
    end
end
center=p;
end

function attr=Norm_attr(Atrr_c,N)
for i=1:size(Atrr_c,2)-1%��һ����ɢ������
    Atrr_c(:,i)=Atrr_c(:,i)-min(Atrr_c(:,i))+1;
    if max(Atrr_c(:,i))>N
       Atrr_c(find(Atrr_c(:,i)==max(Atrr_c(:,i))),i)=N;      
    end
    if max(Atrr_c(:,i))~=length(unique(Atrr_c(:,i)))%���¶����Ե�ȡֵ���б��
       for j=1:length(unique(Atrr_c(:,i)))
           u=unique(Atrr_c(:,i));
           Atrr_c(find(Atrr_c(:,i)==u(j)),i)=j;
       end
    end
end
clear u;
attr=Atrr_c;
end


function dis=Distance(point1,point2,data,n)%����������֮���ʷٻ�������еľ���,dataΪ�������ݼ�,point1,pont2����һ�д���һ��ʵ��,n����ǰn������������������
%���������������ľ���
r_n=zeros(1,size(data,1));
for i=1:size(data,1)%���㵱ǰʵ��
    r_n(i)=norm(point1(1:n)-data(i,1:n),2);
end
r1=n/(size(data,2))*(norm(point1(1:n)-point2(1:n),2)-min(r_n))/(max(r_n)-min(r_n));%���������ݵľ���
%��������������ľ���
r_c=zeros(1,size(data,1));
for i=1:size(data,1)%���㵱ǰʵ��
    for j=n+1:size(data,2)
        if point1(j)~=data(i,j)
           r_c(i)=r_c(i)+1;%����ֵ����Ⱦ����1
        end
    end
end
if max(r_c)==min(r_c)
    r2=0;
else
   r2=(size(data,2)-n)/(size(data,2))*(sum((point1~=point2)))/(max(r_c)-min(r_c));%�����������ݵ�֮����������Եľ���
end
dis=r1+r2;
end

function result = Neighborhoodlle(data,d,k,radius)%�������������ݽ��н�ά
[result_n,distance]=Neighborhood(data,radius);%����ÿ�����ݵ�����͸�������֮��ľ���
[q,ind]=sort(distance,2);%��Neighborhoodÿһ�н�����������,aΪ���к��ֵ,indΪ����������ֵ
w=zeros(size(data,1),size(data,1));%����������ʵ����Ȩֵ
k_Neighborhood=zeros(size(data,1),k);%���������������k��ʵ���ı��
for i=1:size(data,1)%ÿ��ʵ��
    for j=2:k+1%ÿһ������,q�е�һ������Ϊ�䱾��
        k_Neighborhood(i,j)=ind(i,j);
        w(i,ind(i,j))=length(intersect(find(result_n(i,:)==1),find(result_n(ind(i,j),:)==1)))/length(union(find(result_n(i,:)==1),find(result_n(ind(i,j),:)==1)));%�����������Ȩֵ
    end
end
%ͶӰ����,�ο�https://www.cnblogs.com/pinard/p/6266408.html?utm_source=itdadao&utm_medium=referral(ע��:�����Ƶ��д���֮��)
I=eye(size(data,1),size(data,1));%���ɵ�λ����
M=(I-w)'*(I-w);%�������M
[v,value]=eig(M);%������������ֵ����������
eigenvalue=diag(value)';%��ȡ���������ֵ
[q,ind]=sort(eigenvalue);%������ֵ����С�����������
ind=ind(2:(d+1));%��һ����С������ֵΪ0,��ȡd����С������ֵ��Ӧ�������������
result=v(:,ind);%��ȡ���յ���������,�õ���ά
end

function [result,distance] = Neighborhood(data,radius)
%����ÿһ�����������
result=zeros(size(data,1),size(data,1));%1��ʾ��������0��ʾ��������
distance=zeros(size(data,1),size(data,1));%����������������֮���ŷʽ����
for i=1:size(data,1)
    for j=i:size(data,1)
        dist= norm(data(i,:)-data(j,:),2);%�������������ľ���
        distance(i,j)=dist;
        distance(j,i)=dist;
        if dist<=radius%����λ��������
            result(i,j)=1;
            result(j,i)=1;
        end
    end
end
end

function result = FeatureSelection(data)
%�������������,ѡ�����Ч������,����������ֵԽСԽ��(����+����֮��������)
mydata=data;%��ֹ��ԭ���ݽ����޸�
label=1;%ָʾ�㷨�Ƿ���ֹ
while label==1
   value_del=zeros(1,size(data,2));
   value=Granular_Sim(mydata);%���㵱ǰ����������������ֵ
   for i=1:size(mydata,2)
       thisdata=mydata;
       thisdata(:,i)=[];%ɾ����ǰ����
       if size(thisdata,2)==0
           thisdata=mydata;
       end
       value_del(i)=Granular_Sim(thisdata);%ɾ����ǰ���Ժ����������ֵ
   end
   [max_value,num]=max(value_del);
  if value<max_value||size(data,2)==1%ɾ�������޷�������ݵ���������
      label=0;%�㷨��ֹ
  else%num��Ӧ������Ϊ������������ɾ��
      data(:,num)=[];
  end
end
result=data;
end

function value = Granular_Sim(data)%�������ݵ��������ƶ�
num=zeros(1,size(data,2));
for i=1:size(data,2)
    num(i)=length(unique(data(:,i)));
end

partition=cell(size(data,2),max(num));
for i=1:size(data,2)%�������ݵ�ÿһά
    for j=1:size(data,1)%����ÿһ������
        partition{i,data(j,i)}=[partition{i,data(j,i)},j];%�γɻ��֣������ﱣ��������ݵ���ţ�ÿһ�м���ÿһά�Ļ���
    end  
end
clear num;
dis_partition=partition(1,:);
dis_partition=delete_empty(dis_partition);
for i=2:size(partition,1)
    tempdata=SetIntersection(dis_partition,delete_empty(partition(i,:)));
    dis_partition=tempdata;
end
value=0;
for i=2:size(dis_partition,2)%���㻮�ֽ��������
    value=value+(1/(size(data,1)^2))*length(dis_partition{i})^2;
end
value=value*1/size(data,1);
%�������ά��֮������ƶ�
sim_value=0;
for i=1:size(data,2)%����ÿһ��ά��
    for j=i+1:size(data,2)
        s=Sim(partition(i,1),partition(j,1));
        sim_value=sim_value+s;
    end
end
sim_value=1/(size(data,2)^2)*sim_value;
value=value+sim_value;
end

function result = delete_empty(dis_partition)
num=[];
for i=1:size(dis_partition,2)%ɾ������
    if isempty(dis_partition{i})==1
       num=[num,i];
    end
end
dis_partition(num)=[];
result=dis_partition;
end


function result = SetIntersection(data1,data2)
%����data1��data2�Ľ���,����data1��data2Ϊ��ά����
result={};%��ʼ�����
for i=1:size(data1,2)
    for j=1:size(data2,2)
        temp=intersect(data1{1,i},data2{1,j});
        result=[result,temp];
        clear temp;
    end
end
%result(1)=[];
%ȥ���ظ��Ľ���
for i=1:size(result,2)-1 
   for j=(i+1):size(result,2)
       if isequal(result{i},result{j})==1
          result(j)=[];
          i=1;
          break;
       end
   end
end
end

function s = Sim(data1,data2)%����data1,data2������������Ӧά�ȵ����ƶ�
    s=0;
    if length(data1)<=length(data2)%data1��Ӧ��ά���γɵĻ��ֿ���ĿС��data2��Ӧ��ά���γɵĻ��ֿ�
        for i=1:length(data1)%ÿһ��ѡ����һ���ָ��Ƕ����Ŀ�������ƶ�
            s_value=0;
            for j=1:length(data2)%Ѱ�Ҹ��Ƕ����Ŀ�
                s_temp=length(intersect(data1{i},data2{j}))/length(union(data1{i},data2{j}));
                if s_temp>s_value
                   s_value=s_temp;
                end
            end
            s=s+s_value;%��¼��ǰ������ƶ�
        end
    else
        for i=1:length(data2)%ÿһ��ѡ����һ���ָ��Ƕ����Ŀ�������ƶ�
            s_value=0;
            for j=1:length(data1)%Ѱ�Ҹ��Ƕ����Ŀ�
                s_temp=length(intersect(data1{i},data2{j}))/length(union(data1{i},data2{j}));
                if s_temp>s_value
                   s_value=s_temp;
                end
            end
            s=s+s_value;%��¼��ǰ������ƶ�
        end
    end
end
