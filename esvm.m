function [TrainingTime, TrainingAccuracy, TrainingSensitifity, TrainingSpecificity, TrainingAUC, TestingTime, TestingAccuracy, TestingSensitifity, TestingSpecificity, TestingAUC]= esvm(TrainingData_File, TestingData_File, C, NumberofHiddenNeurons, ActivationFunction)

% Extreme Support Vector Machine (ESVM) merupakan metode SVM yang memanfaatkan ELM sebagai feature mapping. 
% Metode optimasi yang digunakan untuk mencari hyperplane yang memisahkan dua kelas yang berbeda dalam suatu feature space 
% 
% Input:
% TrainingData_File     - Nama file data training
% TestingData_File      - Nama file data testing
% NumberofHiddenNeurons - Jummlah Hidden neuron yang digunakan
% ActivationFunction    - Type of activation function:
%                           'sig' for Sigmoidal function
%                           'sin' for Sine function
%                           'hardlim' for Hardlim function
%                           'tribas' for Triangular basis function
%                           'radbas' for Radial basis function (for additive type of SLFNs instead of RBF type of SLFNs)
% C                     - Constraint
% Output: 
% TrainingTime          - Time (seconds) spent on training ELM
% TestingTime           - Time (seconds) spent on predicting ALL testing data
% TrainingAccuracy 
% TrainingSensitifity
% TrainingSpecificity
% TrainingAUC
% TrainingTime
% TestingAccuracy
% TestingSensitifity
% TestingSpecificity
% TestingAUC
% TestingTime
% 
% Contoh cara memanggila fungsi ESVM
% -----------------------------------------------------------------------------------------------
% -> [TrainingTime, TrainingAccuracy, TrainingSensitifity, TrainingSpecificity, TrainingAUC,...<-
% -> TestingTime, TestingAccuracy, TestingSensitifity, TestingSpecificity, TestingAUC] ...     <-
% -> = esvm(training, testing, 900, 'sig',350);                                                <-
% -----------------------------------------------------------------------------------------------

%%%%%%%%%%% Load training dataset
% train_data=load(TrainingData_File);
train_data=TrainingData_File;
clear TrainingData_File;
T=train_data(:,end)';
P=train_data(:,1:end-1)';
clear train_data;                                   %   Release raw training data array

%%%%%%%%%%% Load testing dataset
% test_data=load(TestingData_File);
test_data=TestingData_File;
clear TestingData_File;
TV.T=test_data(:,end)';
TV.P=test_data(:,1:end-1)';
clear test_data;                                    %   Release raw testing data array

NumberofTrainingData=size(P,2);
NumberofTestingData=size(TV.P,2);
NumberofInputNeurons=size(P,1);
D = diag(T);

%%%%%%%%%%% Calculate weights & biases

%%%%%%%%%%% Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
InputWeight=rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
BiasofHiddenNeurons=rand(NumberofHiddenNeurons,1);
tempH=InputWeight*P;
clear P;                                            %   Release input of training data 
ind=ones(1,NumberofTrainingData);
e = ind';
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH=tempH+BiasMatrix;
start_time_train=cputime;
%%%%%%%%%%% Calculate hidden neuron output matrix H
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-tempH));
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = double(hardlim(tempH));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H = tribas(tempH);
    case {'radbas'}
        %%%%%%%% Radial basis function
        H = radbas(tempH);
        %%%%%%%% More activation functions can be added here                
end
clear tempH;                                        %   Release the temparary array for calculation of hidden neuron output matrix H
H=[H;-ones(1,NumberofTrainingData)];
%%%%%%%%%%% Calculate output weights OutputWeight (beta_i)

OutputWeight=(eye(size(H,1))/C+H * H') \ H * (D * e); %change this with equation 15 like hournal ESVM by Q. Liu etc     

%%% if size(H,2) < size(H,1)
%%%     OutputWeight = H / (eye(size(H,2)) / C + H' * H) * (D * e);
%%% end

%If you use faster methods or kernel method, PLEASE CITE in your paper properly: 

%%%%%%%%%%% Calculate the training accuracy
Y=sign((H' * OutputWeight)'); 
end_time_train=cputime;
TrainingTime=end_time_train-start_time_train; 
%   Y: the actual output of the training data
% [TrainingAccuracy, TrainingSensitifity, TrainingSpecificity, TrainingAUC] =  confusionMatrix(T,Y) ;              %   Calculate training accuracy for classification case
[TrainingAccuracy, TrainingSensitifity, TrainingSpecificity, TrainingAUC] =  confusionMatrix(T,Y) ;
clear H;

%%%%%%%%%%% Calculate the output of testing input

tempH_test=InputWeight*TV.P;
clear TV.P;             %   Release input of testing data             
ind=ones(1,NumberofTestingData);
BiasMatrix=BiasofHiddenNeurons(:,ind);              %   Extend the bias matrix BiasofHiddenNeurons to match the demention of H
tempH_test=tempH_test + BiasMatrix;
switch lower(ActivationFunction)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H_test = 1 ./ (1 + exp(-tempH_test));
    case {'sin','sine'}
        %%%%%%%% Sine
        H_test = sin(tempH_test);        
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H_test = hardlim(tempH_test);        
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H_test = tribas(tempH_test);        
    case {'radbas'}
        %%%%%%%% Radial basis function
        H_test = radbas(tempH_test);        
        %%%%%%%% More activation functions can be added here        
end
H_test=[H_test;-ones(1,NumberofTestingData)];
TY=sign((H_test' * OutputWeight)');                       %   TY: the actual output of the testing data
         %   Calculate CPU time (seconds) spent by ELM predicting the whole testing data
start_time_test=cputime;
[TestingAccuracy, TestingSensitifity, TestingSpecificity, TestingAUC] = confusionMatrix(TV.T,TY) ;
end_time_test=cputime;
TestingTime=end_time_test-start_time_test ; 
% [TestingAccuracy, TestingSensitifity, TestingSpecificity, TestingAUC] = confusionMatrix(TV.T,TY) ;     %   Calculate testing accuracy for classification case
      
