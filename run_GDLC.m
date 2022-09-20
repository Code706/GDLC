
clc
clear;

% load('JAFFE_64x64');

%To process the datasets orlraws10P,BASEHOCK,PCMAC,SMK_CAN_187, add the following code
load('BASEHOCK');
fea=X; 
gnd=Y;

fea = im2double(fea);
nClass = length(unique(gnd));
fea = NormalizeFea(fea);
[LABEL, ACC, NMI, obj_NMF] = GDLC(fea', nClass, gnd);



