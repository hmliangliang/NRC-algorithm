function [ R,FM,P,K,RT] = Evaluation(result,Label)
%本函数主要计算四种度量指标Rand statistic, Fowlkes and Mallows index, Purity, Normalized Mutual Information
%输入  result:聚类算法的类簇,数据结构为cell型     Label:数据的真实类标签
%输出   R: Rand statistic  FM: Fowlkes and Mallows index  P: Purity
%初始化各种输出参数的值
R=0;
FM=0;
P=0;
%计算R和FM指标值
SS=0;
SD=0;
DS=0;
DD=0;
Re=zeros(size(Label,1),1);%将result的cell型转化成数组型
for i=1:size(result,2)
    Re(result{i},1)=i;
end
for i=1:size(Re,1)-1
    for j=i+1:size(Re,1)
        if Re(i,1)==Re(j,1)%两个对象处于同一个类簇中
            if Label(i,1)==Label(j,1)%两个对象的类标签也相等
                SS=SS+1;
            else%两个对象的类标签不相等
                SD=SD+1;
            end
        else%两个对象处于不同的类簇中
            if Label(i,1)==Label(j,1)%两个对象的类标签相等
                DS=DS+1;
            else%两个对象的类标签也不相等
                DD=DD+1;
            end
        end
    end
end
R=(SS+DD)/(SS+DD+SD+DS);%获得R值
FM=sqrt(SS*SS/((SS+DS)*(SS+SD)));%获得FM值
%将Label转换成cell型
A=(unique(Label))';%A为保存的是原样本的类标签的个数
k=length(unique(Label));%真实的类标签个数
La=cell(1,k);
for i=1:k
    La{i}=(find(Label(:,1)==A(i)))';%找到同一类的样本序号,并把序号保存到相应的数组元素中去
end
%计算朴素度
for i=1:size(result,2)
    number=0;
    d=zeros(1,k);%保存每一个类的交集元素的个数
    for j=1:k
        d(j)=length(intersect(result{i},La{j}));
    end
    number=max(number,max(d));
    P=P+number/size(Label,1);%获取朴素度的值
end
K=0.5*(SS/(SS+SD)+SS/(SS+DS));
RT=2*SS/(2*SS+DS+SD);
end

