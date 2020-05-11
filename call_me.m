clc
clear all
data=readtable('EEG.csv', 'HeaderLines',1);
data=table2array(data);
data=[data(:,1:end-1),data(:,end)];
[training,testing] = holdout(data,80);
for c=[1:10]
% for C=[0.001,0.005 0.01, 0.025, 0.1, 0.5, 1, 10, 50, 100]
%     for neuron=[ 1, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
[TrainingTime, TrainingAccuracy, TrainingSensitifity, TrainingSpecificity, TrainingAUC,...
TestingTime, TestingAccuracy, TestingSensitifity, TestingSpecificity, TestingAUC] ...
= esvm(training, testing, 0.01, 1000,'sig');
[TrainingAccuracy, TrainingSensitifity, TrainingSpecificity, TrainingAUC, TrainingTime, ...
TestingAccuracy, TestingSensitifity, TestingSpecificity, TestingAUC, TestingTime]
%     end
% [TestingAccuracy, TestingAUC, TestingTime]
end
