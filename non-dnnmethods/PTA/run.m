%% start
clc;
clear;
close all;
addpath(genpath('../../data/')); 
addpath(genpath('./inexact_alm_rpca'));
key = 'porcine4_174x130x31';
save_dir=['D:\dev\experimental_results\detector_outputs\PTA\'];
if ~isfolder(save_dir)
    mkdir(save_dir);
end

%% load data
disp(key)
input=[key,'.mat'];
hsi = load(input);
DataTest = double(hsi.data);
mask = double(hsi.map);
[H,W,Dim]=size(DataTest);
num=H*W;
for i=1:Dim % norm
    DataTest(:,:,i) = (DataTest(:,:,i)-min(min(DataTest(:,:,i)))) / (max(max(DataTest(:,:,i))-min(min(DataTest(:,:,i)))));
end

%% data prepare
mask_reshape = reshape(mask, 1, num);
anomaly_map = logical(double(mask_reshape)>0);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
normal_map = logical(double(mask_reshape)==0);
Y=reshape(DataTest, num, Dim)';

%% Optimal Parameters       
% param:rank r
truncate_rank = 5; % gulfport 

% param:μ
mu = 0.1; % gulfport 

alphia=1;
beta=1e-2;
tau=1;

%% PTA Method
disp('Running PTA, Please wait...')
tol1=1e-4; % 无用
tol2=1e-6; % ε停止迭代的条件
maxiter=400;


tic;
[X,S,area] = AD_Tensor_LILU1(DataTest,alphia,beta,tau,mu,truncate_rank,maxiter,tol1,tol2,normal_map,anomaly_map);
toc

show=sqrt(sum(S.^2,3));
show=(show-min(show(:)))/(max(show(:))-min(show(:)));


save([save_dir,'PTA_map.mat'],'show')

