%% Start
clc;
clear;
close all;
addpath(genpath('../../data/'));
key = 'porcine4_696x520x31';
save_dir=['../../results/', key, '/'];
if ~isfolder(save_dir)
    mkdir(save_dir);
end
addpath(genpath('./Kernel'));

%% Load data
disp(key)
input=[key,'.mat'];
load(input)
data1 = data;
[row, col, bands] = size(data1);
data2 = NormalizeData(data1);
tic
%% Set default parameters
disp('Running KIFD, Please wait...')
zeta = 300; % param:ζ 
if zeta < bands
    % Perform KPCA
    disp(['Performing KPCA: reducing ', num2str(bands), ' → ', num2str(zeta), ' components...']);
    data_kpca = kpca(data2, 10000, zeta, 'Gaussian', 1);
    data = NormalizeData(data_kpca);
else
    % Skip KPCA
    disp(['Skipping KPCA because zeta (', num2str(zeta), ...
          ') >= number of bands (', num2str(bands), ').']);
    data = data2;  
end
data = ToVector(data);

tree_num = 1000;
subsample_percentage = 3;
% subsample_percentage = 6;
% tree_num = 2000;

tree_size = floor(subsample_percentage * row * col /100); 

%% Run global iForest
s = iforest(data, tree_num, tree_size); % 1 hyperspectral data  2 number of isolation  3 trees subsample size
%% Run local iForest iteratively
img = reshape(s, row, col);
stop_flag = 0;
index = [];
num = 1;
r0 = img;
lev = graythresh(r0);   %

if strcmp(key, "porcine1_696x520x31")
    a = 1900;
    %a = floor(row*col/120)
elseif strcmp(key, "porcine2_1392x1040x31")
    a = 10000;
elseif strcmp(key, 'porcine2_696x520x31')
    a = 5000;
elseif strcmp(key, "porcine3_1392x1040x31")
    a = 14500;
elseif strcmp(key, "porcine3_696x520x31")
    a = 7250;
elseif strcmp(key, "porcine4_1392x1040x31")
    a = 7500;
elseif strcmp(key, "porcine4_696x520x31")
    a = 3750;
else
    % fallback / default case
    a = floor(row*col/120);
end

while stop_flag == 0
    [r1, flag, s1, index1] = Local_iforest(r0, data, s, index, lev, a); 
    r0 = r1;
    s = s1;
    index = index1;
    stop_flag = flag;
    num = num + 1;
    if num > 100 
        break;
    end
end
img = zeros(row,col);
img(index1) = 1;
index = (1:row*col)';
index(index1, :) = [];
Data_d = data(:, :);
Data_d(index1,:) = [];
s_d = iforest(Data_d, tree_num, tree_size); 
r1(index) = s_d;
r2 = 10.^r1;
%% Evaluate the results
toc
show = mat2gray(r2);

save([save_dir,'KIFD_map.mat'],'show')