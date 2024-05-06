clear all;
close all;

import tds_control.stability.compute_gamma_r

n=4
A0 = [[-1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, -10, -4],
      [0, 0, 4, -10]]
A1 = [[3, 3, 3, 3],
      [0, -1.5, 0, 0],
      [0, 0, 3, -5],
      [0, 5, 5, 5]]
E = eye(n)
A = {A0, A1}
hA = [0, 1]

r = -1.5

%% Create DDAE
rdde = tds_create(A, hA);
rdde.E % has to be identity
rdde.uE

%%
tds_gamma_r(rdde, -1.5)
