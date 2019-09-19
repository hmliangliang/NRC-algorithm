function [ final_data,final_target,MSE_value,center,mse] = NCluster( data,d,k,k_cluster,N_max)%main function
%UNTITLED Summary of this function goes here
%data为数据,每一行代表一个实例
%k、d主要用在lle算法中,k表示KNN中的k值,d为lle算法降维后数据的维度
%radius针对lle算法中领域的半径,主要用在连续型数据
%k_cluester为聚类结果类簇的数目
%采用领域粗糙集模型来对数据进行聚类
%radius2为聚类过程中,邻域粗糙集邻域的半径
%N_max为聚类算法迭代的次数
%在输出参数表示的含义如下：
%final_target:保存最后的聚类结果对应真实类标签,每一行代表一个实例,
%MSE_value:每一次聚类结果的距离误差
%cluster_result:记录每一次聚类的结果,cluster_result(x).result(i,j)=1表示在x次聚类中,第j个实例划归到第i个类簇中
%center为聚类中心点
col=size(data,2);
%data=data(:,1:(col-1));%数据
%分类型属性与连续型属性
Atrr_c=[];%分类型属性
Atrr_n=[];%连续型属性
lamba=0.9;%计算中心点下近似集所占的比重
N=20;%区分属性是属于连续型还是分类型属性值的数目
for i=1:col
   if length(unique(data(:,i)))<=N%说明该属性为分类型
      Atrr_c=[Atrr_c,data(:,i)];
   else%说明该属性为数值型
      Atrr_n=[Atrr_n,data(:,i)];
   end
end
if isempty(Atrr_n)==1%没有数值型属性
    Atrr_n=[Atrr_n,rand(size(data,1),2)];
end
%d=round(0.3*size(Atrr_n,2));%确定新型lle算法的d值
%disp(['新型lle算法降维后的维度的数目d为:d=',num2str(d)]);
Atrr_c=round(Atrr_c);
Atrr_c=Norm_attr(Atrr_c,N);
%对分类型属性进行特征选择,Atrr_cf保存选择后的特征数据
Atrr_cf=FeatureSelection(Atrr_c);
%disp(['算法降维后的维度的数目为:',num2str(d+size(Atrr_cf,2))]);
%对数值型数据进行归一化
Atrr_n=zscore(Atrr_n);%进行标准化
eu=randi(size(data,1),1,11);
d1=[];
d2=[];
for i=2:11
    d1=[d1,norm( Atrr_n(eu(1),:)- Atrr_n(eu(i),:))];
    d2=[d2,sum(Atrr_c(eu(1),:)~=Atrr_c(eu(i),:))];
end
radius=mean(d1);
radius2=round(mean(d2))+radius;
%disp(['数值型邻域阈值radius=',num2str(radius)]);
%disp(['分类型+数值型邻域阈值radius2=',num2str(radius2)]);
clear d1;
clear d2;
%[y,Ps]=mapminmax(Atrr_n',0,1);%y保存归一化后的结果
%对连续型数据进行降维
y=Atrr_n';
mydata_n=Neighborhoodlle(y',d,k,radius);
data=[mydata_n,Atrr_cf];
n=size(mydata_n,2);%前n个特征为连续型特征
final_data=data;
%计算每个数据对象的邻域集合
neighbor=zeros(size(data,1),size(data,1));%保存邻域对象
for i=1:size(data,1)
    for j=i:size(data,1)
        d_distace=n/(size(data,2))*norm(data(i,1:n)-data(j,1:n),2)+(size(data,2)-n)/(size(data,2))*sum((data(i,1+n:size(data,2))~=data(j,n+1:size(data,2))));%计算两者之间的距离
        if d_distace<=radius2%第j个元素属于第i个元素的邻域
            neighbor(i,j)=1;
        end 
    end
end
%对数据进行聚类
p=data(randi(size(data,1),1,k_cluster),:);%随机选取种子点，每一行代表一个种子点
%cluster_result=struct('result',zeros(k_cluster,size(data,1)));%记录每一次聚类的结果,cluster_result(x).result(i,j)=1表示在x次聚类中,第j个实例划归到第i个类簇中
MSE_value=[];%保存每一次聚类结果的距离误差
cluster_result=zeros(k_cluster,size(data,1));%记录最终的聚类结果
for i=1:N_max
   %1.计算中心点的邻域集合
   p1=p;
   result=zeros(k_cluster,size(data,1));%记录每一种子点的邻域,result(i,j)=1表示第j个实例属于第i个种子点的领域
   for j=1:k_cluster
       for x=1:size(data,1)
           d=n/(size(data,2))*norm(p1(j,1:n)-data(x,1:n),2)+(size(data,2)-n)/(size(data,2))*sum((p1(j,1+n:size(data,2))~=data(x,n+1:size(data,2))));%计算两者之间的距离
           if d<=radius2 %属于邻域内
               result(j,x)=1;
           end
       end
   end
   %计算上近似集与下近似集
   low_approximate=cell(1,k_cluster);
   upper_approximate=cell(1,k_cluster);
   for j=1:k_cluster
       for x=1:size(data,1)
           if all(ismember(find(neighbor(x,:)==1),find(result(j,:)==1)))==1%数据属于子集,属于上近似集
               low_approximate{j}=[low_approximate{j},x];
           elseif isempty(intersect(find(neighbor(x,:))==1,find(result(j,:)==1)))==0%属于下近似集
               upper_approximate{j}=[upper_approximate{j},x];
           end
       end
       upper_approximate{j}=[upper_approximate{j},low_approximate{j}];
       %消除重复的元素
       upper_approximate{j}=unique(upper_approximate{j});
       low_approximate{j}=unique(low_approximate{j});
       %更新中心点
       p1(j,:)=lamba*mean(data(low_approximate{j},:))+(1-lamba)*mean(data(upper_approximate{j},:));
       p1(j,:)=[p1(j,1:n),round(p1(j,n+1:size(data,2)))];%获得最终的聚类中心
   end
   %判断是否达到终止条件
   %计算评价指标
   mse=0;%计算中心点的变化距离
   for j=1:k_cluster
       rr=n/(size(data,2))*norm(p(j,1:n)-p1(j,1:n),2)+(size(data,2)-n)/(size(data,2))*sum(p(j,n+1:size(data,2))~=p1(j,1+n:size(data,2)));
       mse=mse+rr;
       MSE_value=[MSE_value,mse];
   end
   if mse>radius/5%中心点变化较大,积累过程未完成，需要更新中心点
       p=p1;
       clear p1;
   else%聚类过程完成,
       break;
   end
end
%确定每一个数据的归属的类簇
for c=1:size(data,1)
    rt=zeros(1,k_cluster);
    for j=1:k_cluster
        rt(j)=Distance(p(j,:),data(c,:),data,n);
    end
    [t,y]=min(rt);%获取最终的聚类结果,t为最小的距离值,y为序号
    cluster_result(y,c)=1;%确定数据的归属
end
%把聚类结果转换成原来的类标签
final_result=cluster_result;%获取最后一次的聚类结果，每一个行代表一个类簇
%计算均方误差
mse=0;
for j=1:size(final_result,1)
     label=find(final_result(j,:)>0);
     for x=1:length(label)
         tempdata=data(label(x),:);
         mse=mse+norm(tempdata(1:n)-p(j,1:n),2)+sum(tempdata(n+1:size(data,2))~=p(j,1+n:size(data,2)));
     end
end
mse=1/size(data,1)*mse;

final_target=cell(1,k_cluster);%保存最后的聚类结果，每一行代表一个实例
for j=1:k_cluster
%     disp(['j=',num2str(j)]);
%     find(final_result(j,:)==1)
    if isempty(find(final_result(j,:)==1))==0
       final_target{1,j}=find(final_result(j,:)==1);%转换成原来对应的类标签
    end
end
center=p;
end

function attr=Norm_attr(Atrr_c,N)
for i=1:size(Atrr_c,2)-1%归一化离散型属性
    Atrr_c(:,i)=Atrr_c(:,i)-min(Atrr_c(:,i))+1;
    if max(Atrr_c(:,i))>N
       Atrr_c(find(Atrr_c(:,i)==max(Atrr_c(:,i))),i)=N;      
    end
    if max(Atrr_c(:,i))~=length(unique(Atrr_c(:,i)))%重新对属性的取值进行编号
       for j=1:length(unique(Atrr_c(:,i)))
           u=unique(Atrr_c(:,i));
           Atrr_c(find(Atrr_c(:,i)==u(j)),i)=j;
       end
    end
end
clear u;
attr=Atrr_c;
end


function dis=Distance(point1,point2,data,n)%计算两个点之间的史倩玉论文中的距离,data为整个数据集,point1,pont2均是一行代表一个实例,n代表前n个特征是连续型数据
%计算连续型特征的距离
r_n=zeros(1,size(data,1));
for i=1:size(data,1)%计算当前实例
    r_n(i)=norm(point1(1:n)-data(i,1:n),2);
end
r1=n/(size(data,2))*(norm(point1(1:n)-point2(1:n),2)-min(r_n))/(max(r_n)-min(r_n));%连续型数据的距离
%计算分类型特征的距离
r_c=zeros(1,size(data,1));
for i=1:size(data,1)%计算当前实例
    for j=n+1:size(data,2)
        if point1(j)~=data(i,j)
           r_c(i)=r_c(i)+1;%属性值不相等距离加1
        end
    end
end
if max(r_c)==min(r_c)
    r2=0;
else
   r2=(size(data,2)-n)/(size(data,2))*(sum((point1~=point2)))/(max(r_c)-min(r_c));%计算两个数据点之间分类性属性的距离
end
dis=r1+r2;
end

function result = Neighborhoodlle(data,d,k,radius)%对于连续型数据进行降维
[result_n,distance]=Neighborhood(data,radius);%计算每个数据的邻域和各个对象之间的距离
[q,ind]=sort(distance,2);%对Neighborhood每一行进行升序排列,a为排列后的值,ind为排序后的索引值
w=zeros(size(data,1),size(data,1));%保存邻域内实例的权值
k_Neighborhood=zeros(size(data,1),k);%保存邻域内最近的k个实例的标号
for i=1:size(data,1)%每个实例
    for j=2:k+1%每一个邻域,q中第一个样本为其本身
        k_Neighborhood(i,j)=ind(i,j);
        w(i,ind(i,j))=length(intersect(find(result_n(i,:)==1),find(result_n(ind(i,j),:)==1)))/length(union(find(result_n(i,:)==1),find(result_n(ind(i,j),:)==1)));%根据邻域计算权值
    end
end
%投影计算,参考https://www.cnblogs.com/pinard/p/6266408.html?utm_source=itdadao&utm_medium=referral(注意:博客推导有错误之处)
I=eye(size(data,1),size(data,1));%生成单位矩阵
M=(I-w)'*(I-w);%计算矩阵M
[v,value]=eig(M);%计算矩阵的特征值与特征向量
eigenvalue=diag(value)';%提取矩阵的特征值
[q,ind]=sort(eigenvalue);%对特征值按由小到大进行排序
ind=ind(2:(d+1));%第一个最小的特征值为0,获取d个最小的特征值对应的特征向量序号
result=v(:,ind);%获取最终的特征向量,得到降维
end

function [result,distance] = Neighborhood(data,radius)
%返回每一个对象的领域
result=zeros(size(data,1),size(data,1));%1表示属于邻域，0表示不是邻域
distance=zeros(size(data,1),size(data,1));%保存任意两个对象之间的欧式距离
for i=1:size(data,1)
    for j=i:size(data,1)
        dist= norm(data(i,:)-data(j,:),2);%计算两个对象间的距离
        distance(i,j)=dist;
        distance(j,i)=dist;
        if dist<=radius%距离位于领域内
            result(i,j)=1;
            result(j,i)=1;
        end
    end
end
end

function result = FeatureSelection(data)
%处理分类型数据,选择出有效的特征,特征的评价值越小越好(粒度+特征之间的相关性)
mydata=data;%防止对原数据进行修改
label=1;%指示算法是否终止
while label==1
   value_del=zeros(1,size(data,2));
   value=Granular_Sim(mydata);%计算当前数据所有特征评价值
   for i=1:size(mydata,2)
       thisdata=mydata;
       thisdata(:,i)=[];%删除当前属性
       if size(thisdata,2)==0
           thisdata=mydata;
       end
       value_del(i)=Granular_Sim(thisdata);%删除当前属性后的特征评估值
   end
   [max_value,num]=max(value_del);
  if value<max_value||size(data,2)==1%删除特征无法提高数据的区分能力
      label=0;%算法终止
  else%num对应的特征为冗余特征可以删除
      data(:,num)=[];
  end
end
result=data;
end

function value = Granular_Sim(data)%计算数据的粒度相似度
num=zeros(1,size(data,2));
for i=1:size(data,2)
    num(i)=length(unique(data(:,i)));
end

partition=cell(size(data,2),max(num));
for i=1:size(data,2)%对于数据的每一维
    for j=1:size(data,1)%对于每一个数据
        partition{i,data(j,i)}=[partition{i,data(j,i)},j];%形成划分，划分里保存的是数据的序号，每一行即是每一维的划分
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
for i=2:size(dis_partition,2)%计算划分结果的粒度
    value=value+(1/(size(data,1)^2))*length(dis_partition{i})^2;
end
value=value*1/size(data,1);
%计算各个维度之间的相似度
sim_value=0;
for i=1:size(data,2)%对于每一个维度
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
for i=1:size(dis_partition,2)%删除空项
    if isempty(dis_partition{i})==1
       num=[num,i];
    end
end
dis_partition(num)=[];
result=dis_partition;
end


function result = SetIntersection(data1,data2)
%计算data1与data2的交集,其中data1与data2为二维集合
result={};%初始化结果
for i=1:size(data1,2)
    for j=1:size(data2,2)
        temp=intersect(data1{1,i},data2{1,j});
        result=[result,temp];
        clear temp;
    end
end
%result(1)=[];
%去除重复的交集
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

function s = Sim(data1,data2)%计算data1,data2两个划分所对应维度的相似度
    s=0;
    if length(data1)<=length(data2)%data1对应的维度形成的划分块数目小于data2对应的维度形成的划分块
        for i=1:length(data1)%每一块选择另一划分覆盖度最大的块计算相似度
            s_value=0;
            for j=1:length(data2)%寻找覆盖度最大的块
                s_temp=length(intersect(data1{i},data2{j}))/length(union(data1{i},data2{j}));
                if s_temp>s_value
                   s_value=s_temp;
                end
            end
            s=s+s_value;%记录当前块的相似度
        end
    else
        for i=1:length(data2)%每一块选择另一划分覆盖度最大的块计算相似度
            s_value=0;
            for j=1:length(data1)%寻找覆盖度最大的块
                s_temp=length(intersect(data1{i},data2{j}))/length(union(data1{i},data2{j}));
                if s_temp>s_value
                   s_value=s_temp;
                end
            end
            s=s+s_value;%记录当前块的相似度
        end
    end
end
