%% start
clc;
clear;
close all;
addpath(genpath('../../data/')); 
key = 'bovine5.3_174x128x31';
save_dir=['../../results/', key, '/'];
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
lambda = 0.05;
beta = 0.002;
gamma = 0.02;
K = 7;
P = 30;

%% GTVLRR Method
disp('Running GTVLRR, Please wait...')
tic;

Dict=ConstructionD_lilu(Y,K,P);
display = true;
[X,S] = lrr_tv_manifold(Y,Dict,lambda,beta,gamma,[H,W],display);
toc

r_gtvlrr=sqrt(sum(S.^2));
r_max = max(r_gtvlrr(:));
taus = linspace(0, r_max, 5000);
for index2 = 1:length(taus)
  tau = taus(index2);
  anomaly_map_rx = (r_gtvlrr > tau);
  PF(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
  PD(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
end
show=reshape(r_gtvlrr,[H,W]);
show=(show-min(show(:)))/(max(show(:))-min(show(:)));
area = sum((PF(1:end-1)-PF(2:end)).*(PD(2:end)+PD(1:end-1))/2);
disp(['Auc:',num2str(area)])

save([save_dir,'GTVLRR_map.mat'],'show')
save([save_dir,'GTVLRR_roc.mat'],'PD','PF')


