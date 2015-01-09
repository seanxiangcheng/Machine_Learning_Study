% HW1 part II: k-nearest neighbors
% Ex. 2.8 in The Elements of Statistical Learning
% http://statweb.stanford.edu/~tibs/ElemStatLearn/
clear all
load zip.train;
A=zip;
load zip.test;
B=zip;

L_res=size(A,2)-1;
L_train=size(A,1);
L_test=size(B,1);

[IDX,D]=knnsearch(A(:,2:(L_res+1)),A(:,2:(L_res+1)),'k',16);
A_test_k1=round(mode(reshape(A(IDX(:,2),1),L_train,1),2));
A_test_k3=round(mode(reshape(A(IDX(:,2:4),1),L_train,3),2));
A_test_k5=round(mode(reshape(A(IDX(:,2:6),1),L_train,5),2));
A_test_k7=round(mode(reshape(A(IDX(:,2:8),1),L_train,7),2));
A_test_k15=round(mode(reshape(A(IDX(:,2:16),1),L_train,15),2));


IDX=knnsearch(A(:,2:(L_res+1)),B(:,2:(L_res+1)),'k',16);
B_test_k1=round(mode(reshape(A(IDX(:,2),1),L_test,1),2));
B_test_k3=round(mode(reshape(A(IDX(:,2:4),1),L_test,3),2));
B_test_k5=round(mode(reshape(A(IDX(:,2:6),1),L_test,5),2));
B_test_k7=round(mode(reshape(A(IDX(:,2:8),1),L_test,7),2));
B_test_k15=round(mode(reshape(A(IDX(:,2:16),1),L_test,15),2));

%%%%%%%%%%%%%%%% Simple data analysis & output %%%%%%%%%%%%%%%%%%%
A_Error_num=nnz(A_test_k1-A(:,1));
Ar2=find(A(:,1)==2);
A2_Error_num=nnz(2-A_test_k1(Ar2));
Ar3=find(A(:,1)==3);
A3_Error_num=nnz(3-A_test_k1(Ar3));

B_Error_num=nnz(B_test_k1-B(:,1));
Br2=find(B(:,1)==2);
B2_Error_num=nnz(2-B_test_k1(Br2));
Br3=find(B(:,1)==3);
B3_Error_num=nnz(3-B_test_k1(Br3));

disp('Total number of trainings:');
disp(L_train);
disp('Total number of tests:');
disp(L_test);

disp(' ');
disp('****** 1-nearest neighbor  ******');
disp('Portion of errors by 1nn in the training set');
disp(A_Error_num/L_train);
disp('Portion of errors by 1nn in the test set');
disp(B_Error_num/L_test);
disp('Portion of errors for 2''s in TrS');
disp(A2_Error_num/length(Ar2));
disp('Portion of errors for 2''s in TeS');
disp(B2_Error_num/length(Br2));
disp('Portion of errors for 3''s in TrS');
disp(A3_Error_num/length(Ar3));
disp('Portion of errors for 3''s in TeS');
disp(B3_Error_num/length(Br3));
disp('****** 1-nearest neighbor  ******');
disp(' ');
disp(' ');


A_Error_num=nnz(A_test_k3-A(:,1));
Ar2=find(A(:,1)==2);
A2_Error_num=nnz(2-A_test_k3(Ar2));
Ar3=find(A(:,1)==3);
A3_Error_num=nnz(3-A_test_k3(Ar3));

B_Error_num=nnz(B_test_k3-B(:,1));
Br2=find(B(:,1)==2);
B2_Error_num=nnz(2-B_test_k3(Br2));
Br3=find(B(:,1)==3);
B3_Error_num=nnz(3-B_test_k3(Br3));

disp(' ');
disp('****** 3-nearest neighbor  ******');
disp('Portion of errors by 1nn in the training set');
disp(A_Error_num/L_train);
disp('Portion of errors by 1nn in the test set');
disp(B_Error_num/L_test);
disp('Portion of errors for 2''s in TrS');
disp(A2_Error_num/length(Ar2));
disp('Portion of errors for 2''s in TeS');
disp(B2_Error_num/length(Br2));
disp('Portion of errors for 3''s in TrS');
disp(A3_Error_num/length(Ar3));
disp('Portion of errors for 3''s in TeS');
disp(B3_Error_num/length(Br3));
disp('****** 3-nearest neighbor  ******');
disp(' ');
disp(' ');



A_Error_num=nnz(A_test_k5-A(:,1));
Ar2=find(A(:,1)==2);
A2_Error_num=nnz(2-A_test_k5(Ar2));
Ar3=find(A(:,1)==3);
A3_Error_num=nnz(3-A_test_k5(Ar3));

B_Error_num=nnz(B_test_k5-B(:,1));
Br2=find(B(:,1)==2);
B2_Error_num=nnz(2-B_test_k5(Br2));
Br3=find(B(:,1)==3);
B3_Error_num=nnz(3-B_test_k5(Br3));

disp(' ');
disp('****** 5-nearest neighbor  ******');
disp('Portion of errors by 1nn in the training set');
disp(A_Error_num/L_train);
disp('Portion of errors by 1nn in the test set');
disp(B_Error_num/L_test);
disp('Portion of errors for 2''s in TrS');
disp(A2_Error_num/length(Ar2));
disp('Portion of errors for 2''s in TeS');
disp(B2_Error_num/length(Br2));
disp('Portion of errors for 3''s in TrS');
disp(A3_Error_num/length(Ar3));
disp('Portion of errors for 3''s in TeS');
disp(B3_Error_num/length(Br3));
disp('****** 5-nearest neighbor  ******');
disp(' ');
disp(' ');


A_Error_num=nnz(A_test_k7-A(:,1));
Ar2=find(A(:,1)==2);
A2_Error_num=nnz(2-A_test_k7(Ar2));
Ar3=find(A(:,1)==3);
A3_Error_num=nnz(3-A_test_k7(Ar3));

B_Error_num=nnz(B_test_k7-B(:,1));
Br2=find(B(:,1)==2);
B2_Error_num=nnz(2-B_test_k7(Br2));
Br3=find(B(:,1)==3);
B3_Error_num=nnz(3-B_test_k7(Br3));

disp(' ');
disp('****** 7-nearest neighbor  ******');
disp('Portion of errors by 1nn in the training set');
disp(A_Error_num/L_train);
disp('Portion of errors by 1nn in the test set');
disp(B_Error_num/L_test);
disp('Portion of errors for 2''s in TrS');
disp(A2_Error_num/length(Ar2));
disp('Portion of errors for 2''s in TeS');
disp(B2_Error_num/length(Br2));
disp('Portion of errors for 3''s in TrS');
disp(A3_Error_num/length(Ar3));
disp('Portion of errors for 3''s in TeS');
disp(B3_Error_num/length(Br3));
disp('****** 7-nearest neighbor  ******');
disp(' ');
disp(' ');


A_Error_num=nnz(A_test_k15-A(:,1));
Ar2=find(A(:,1)==2);
A2_Error_num=nnz(2-A_test_k15(Ar2));
Ar3=find(A(:,1)==3);
A3_Error_num=nnz(3-A_test_k15(Ar3));

B_Error_num=nnz(B_test_k15-B(:,1));
Br2=find(B(:,1)==2);
B2_Error_num=nnz(2-B_test_k15(Br2));
Br3=find(B(:,1)==3);
B3_Error_num=nnz(3-B_test_k15(Br3));

disp(' ');
disp('****** 15-nearest neighbor  ******');
disp('Portion of errors by 1nn in the training set');
disp(A_Error_num/L_train);
disp('Portion of errors by 1nn in the test set');
disp(B_Error_num/L_test);
disp('Portion of errors for 2''s in TrS');
disp(A2_Error_num/length(Ar2));
disp('Portion of errors for 2''s in TeS');
disp(B2_Error_num/length(Br2));
disp('Portion of errors for 3''s in TrS');
disp(A3_Error_num/length(Ar3));
disp('Portion of errors for 3''s in TeS');
disp(B3_Error_num/length(Br3));
disp('****** 15-nearest neighbor  ******');
disp(' ');
disp(' ');
