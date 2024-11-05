clear;

% Replace with name of mat file

FileData = load("test_4_letter_z.mat");

myTable = FileData.ls.Labels.A{99};
allData = myTable{:,:};

% Replace with name of csv file
writematrix(allData,"test_4_letter_z.csv");