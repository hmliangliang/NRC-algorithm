function [ R,FM,P,K,RT] = Evaluation(result,Label)
%��������Ҫ�������ֶ���ָ��Rand statistic, Fowlkes and Mallows index, Purity, Normalized Mutual Information
%����  result:�����㷨�����,���ݽṹΪcell��     Label:���ݵ���ʵ���ǩ
%���   R: Rand statistic  FM: Fowlkes and Mallows index  P: Purity
% NMI: Normalized Mutual Information
%��ʼ���������������ֵ
R=0;
FM=0;
P=0;
%����R��FMָ��ֵ
SS=0;
SD=0;
DS=0;
DD=0;
Re=zeros(size(Label,1),1);%��result��cell��ת����������
for i=1:size(result,2)
    Re(result{i},1)=i;
end
for i=1:size(Re,1)-1
    for j=i+1:size(Re,1)
        if Re(i,1)==Re(j,1)%����������ͬһ�������
            if Label(i,1)==Label(j,1)%������������ǩҲ���
                SS=SS+1;
            else%������������ǩ�����
                SD=SD+1;
            end
        else%���������ڲ�ͬ�������
            if Label(i,1)==Label(j,1)%������������ǩ���
                DS=DS+1;
            else%������������ǩҲ�����
                DD=DD+1;
            end
        end
    end
end
R=(SS+DD)/(SS+DD+SD+DS);%���Rֵ
FM=sqrt(SS*SS/((SS+DS)*(SS+SD)));%���FMֵ
%��Labelת����cell��
A=(unique(Label))';%AΪ�������ԭ���������ǩ�ĸ���
k=length(unique(Label));%��ʵ�����ǩ����
La=cell(1,k);
for i=1:k
    La{i}=(find(Label(:,1)==A(i)))';%�ҵ�ͬһ����������,������ű��浽��Ӧ������Ԫ����ȥ
end
%�������ض�
for i=1:size(result,2)
    number=0;
    d=zeros(1,k);%����ÿһ����Ľ���Ԫ�صĸ���
    for j=1:k
        d(j)=length(intersect(result{i},La{j}));
    end
    number=max(number,max(d));
    P=P+number/size(Label,1);%��ȡ���ضȵ�ֵ
end
K=0.5*(SS/(SS+SD)+SS/(SS+DS));
RT=2*SS/(2*SS+DS+SD);
end

