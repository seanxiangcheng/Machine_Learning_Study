% Machine Learning Homework 5 LDA
% Ex. 4.9 of http://statweb.stanford.edu/~tibs/ElemStatLearn/

clear all
%%%%% Load Trainning data from txt file %%%%%%
disp(' ');
disp('******** Linear Discriminant Analysis ********');
disp('******** Check Results Below ********');
filename = 'train.txt';
delimiterIn=',';
headerlinesIn=1;
A=importdata(filename, delimiterIn, headerlinesIn);

%%%%% Create predictor matrix 'X' and the right classifier vector 'Y'
Y=A.data(:,2);
X=A.data(:,3:12);

tic;
data=tabulate(Y); % find the prior of P(G=1,2,3,...,11) respectively;
Pi=data(:,3)/100; % tabulate() returns the Prior probability in percentage; we convert them into 0<Pi(k)<1

Mu=zeros(length(Pi),size(X,2));
FPi=data(:,2);
for i=1:length(Y)
    Mu(Y(i),:)=Mu(Y(i),:)+X(i,:)/FPi(Y(i));
end
Sigma=zeros(size(X,2));

for i=1:length(Y)
    Sigma=Sigma+(X(i,:)-Mu(Y(i),:))'*(X(i,:)-Mu(Y(i),:));
end
Sigma=Sigma/(length(Y)-length(Pi));
t=toc;
disp('Time spent to train the data:');
disp(t);

%%%%%%%% Test the model using the train data %%%%%%%%%%
tic;
LDFmatrix=zeros(length(Y),length(Pi));
invSigma=inv(Sigma);
for i=1:length(Y)
    for k=1:length(Pi)
        LDFmatrix(i,k)=X(i,:)*invSigma*Mu(k,:)'-0.5*Mu(k,:)*invSigma*Mu(k,:)'+log(Pi(k));
    end
end
t=toc;
[maxvalues,TrainClassPred]=max(LDFmatrix,[],2);
ErrorRate=nnz(TrainClassPred-Y)/length(Y);
disp('Error Rate of training data:');
disp(ErrorRate);
disp('Time spent to test the training data:');
disp(t);


%%%%%%%% Test the model using the test data %%%%%%%%%%
filename = 'test.txt';
delimiterIn=',';
headerlinesIn=1;
A=importdata(filename, delimiterIn, headerlinesIn);
Yt=A.data(:,2);
Xt=A.data(:,3:12);
tic;
LDFmatrix_t=zeros(length(Yt),length(Pi));
invSigma=inv(Sigma);
for i=1:length(Yt)
    for k=1:length(Pi)
        LDFmatrix_t(i,k)=Xt(i,:)*invSigma*Mu(k,:)'-0.5*Mu(k,:)*invSigma*Mu(k,:)'+log(Pi(k));
    end
end
t=toc;
[maxvalues_t,TrainClassPred_t]=max(LDFmatrix_t,[],2);
ErrorRate_t=nnz(TrainClassPred_t-Yt)/length(Yt);
disp('Error Rate of test data:');
disp(ErrorRate_t);
disp('Time spent to test the data:');
disp(t);
disp('******** Check Results Above ********');
disp('******** Linear Discriminant Analysis: Done ********');
disp(' ');
