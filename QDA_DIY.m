% Machine Learning Homework 5 QDA
% Ex. 4.9 of http://statweb.stanford.edu/~tibs/ElemStatLearn/
clear all
%%%%% Load Trainning data from txt file %%%%%%
disp(' ');
disp('******** Qudratic Discriminant Analysis: Running ********');
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
Sigma=zeros(size(X,2),size(X,2),length(Y));

for i=1:length(Y)
    Sigma(:,:,Y(i))=Sigma(:,:,Y(i))+(X(i,:)-Mu(Y(i),:))'*(X(i,:)-Mu(Y(i),:))/(FPi(Y(i))-1);
end
t=toc;
disp('Time spent to train the data:');
disp(t);

%%%%%%%% Test the training data %%%%%%%%%%
tic;
QDFmatrix=zeros(length(Y),length(Pi));
invSigma=Sigma;
detSigma=zeros(length(Pi),1);
for i=1:length(Pi)
    invSigma(:,:,i)=inv(Sigma(:,:,i));
    detSigma(i)=det(Sigma(:,:,i));
end
for i=1:length(Y)
    for k=1:length(Pi)
        QDFmatrix(i,k)=-0.5*log(detSigma(k))-0.5*(X(i,:)-Mu(k,:))*invSigma(:,:,k)*(X(i,:)-Mu(k,:))'+log(Pi(k));
    end
end
t=toc;
[maxvalues,TrainClassPred]=max(QDFmatrix,[],2);
ErrorRate=nnz(TrainClassPred-Y)/length(Y);
disp('Error Rate of training data:');
disp(ErrorRate);
disp('Time spent to test the training data:');
disp(t);


%%%%%%%% Test the model using the Test data%%%%%%%%%%
filename = 'test.txt';
delimiterIn=',';
headerlinesIn=1;
A=importdata(filename, delimiterIn, headerlinesIn);
Yt=A.data(:,2);
Xt=A.data(:,3:12);
tic;
QDFmatrix_t=zeros(length(Yt),length(Pi));
invSigma=Sigma;
detSigma=zeros(length(Pi),1);          
% the calculation of the inverse and determinant could be removed; but we keep it just to compare the speed with LDA
for i=1:length(Pi)
    invSigma(:,:,i)=inv(Sigma(:,:,i));
    detSigma(i)=det(Sigma(:,:,i));
end
for i=1:length(Yt)
    for k=1:length(Pi)
        QDFmatrix_t(i,k)=-0.5*log(detSigma(k))-0.5*(Xt(i,:)-Mu(k,:))*invSigma(:,:,k)*(Xt(i,:)-Mu(k,:))'+log(Pi(k));
    end
end
t=toc;
[maxvalues_t,TrainClassPred_t]=max(QDFmatrix_t,[],2);
ErrorRate_t=nnz(TrainClassPred_t-Yt)/length(Yt);
disp('Error Rate of test data:');
disp(ErrorRate_t);
disp('Time spent to test the data:');
disp(t);
disp('******** Check Results Above ********');
disp('******** Qudratic Discriminant Analysis: Done ********');
disp(' ');
