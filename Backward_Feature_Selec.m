% Backward Feature Selection based on Z score
% The dataset is the South African heart disease data (www-stat.stanford.edu/ElemStatLearn)
% The goal is to repeat the results in the book

clear all
flag=1; 
% flag is 1 is the min(Z)<3, keep dropping the variable with the smallest Z
fprintf('\n****** Backward Stepwise Logistic Regression running ****** \n')
deli=',';
hd = 1;
A=importdata('SAheart.txt',deli,hd);
A.textdata{1}='intercept';
X=A.data(:,2:10);
Y=A.data(:,11)+1; % Y=0 is class One; Y=1 is class Two. 
step=1;
num_var=(1:10);
while flag
fprintf('\n###### Step %d ######\n',step);
step=step+1;
[L,W]=size(X);
[B,dev,stats] = mnrfit(X,Y);
exp_fitted=[ones(L,1),X]*B;
P_fitted=exp(exp_fitted)./(1+exp(exp_fitted));
C_fitted=zeros(L,1);
for i=1:L
    if P_fitted(i)>0.5
        C_fitted(i)=1;
    else
        C_fitted(i)=2;
    end
end
error_rate=nnz(Y-C_fitted)/L;
fprintf('     Using %d Parameters (not including intercept) \n',W);
fprintf('     Error rate is: %-.4f \n',error_rate);
fprintf('                Coefficient | Std Error | Z Score \n');
for i=1:length(B)
    fprintf('     %-12s %-11.3f %-11.3f %-12.3f\n', A.textdata{num_var(i)}, -B(i), stats.se(i), -B(i)/stats.se(i));
end 

%Find the smallest Z score and drop its corresponding variable
[zmin,ind]=min(abs(B./stats.se)); % the 'ind'th varialbe is the 'ind'-1 in X
if zmin>3.0
    flag=0;
elseif ind==2
    X=X(:,2:W);
    num_var=[num_var(1),num_var(3:W+1)];
elseif ind==W+1
    X=X(:,1:W-1);
    num_var=num_var(1:W);
else
    X=[X(:,1:ind-2), X(:,ind:W)];
    num_var=[num_var(1:ind-1), num_var(ind+1:W+1)];
end
end
