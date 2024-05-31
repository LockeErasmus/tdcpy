clear all;
close all;

import tds_control.stability.compute_gamma


A = {0.25, -1/3};
hA = [0, 1];
H = {-0.75, 0.25};
hH = [1,2];

%% Create DDAE
ndde = tds_create_neutral(H, hH, A, hA)
ddae = ndde.get_delay_difference_equation()
%%
tds_gamma_r(ddae, 0)
