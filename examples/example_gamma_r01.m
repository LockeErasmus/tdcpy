clear all;
close all;

% import tds_control.stability.gamma_normalized_diff

% settings 01; gamma0 = 1.0
A = {0.25, -1/3};
hA = [0, 1];
H = {-0.75, 0.25};
hH = [1,2];

% settings 02; gamma0 = 1.8
A = {0.25, -1/3};
hA = [0, 1];
H = {-0.55, 1.25};
hH = [0.1, 0.2];

% settings 03; gamma0 = 7.8903
A = {[0.1, -0.2, -0.5;
      0, 2, -1.0;
      0, -1, -4],
      [0.1, 0.2, -0.5;
      2, 0, -0.8;
      0, -1, -1]};
hA = [0, 1];

H = {[0.1, 0.2, -0.5;
      0, 0, -1.0;
      0, 5, -4],
      [0.1, 0.2, -0.5;
      2, 0, -0.8;
      0, 5, -4],
      [0.1, 1.2, -0.5;
      2, 0, -1.0;
      -2, 5, -4]};
hH = [0.1, 0.2, 0.4];


%% Create DDAE
ndde = tds_create_neutral(H, hH, A, hA)
ddae = ndde.get_delay_difference_equation()
%%

options = tds_gamma_r_options;
options.quiet = false; options.Ntheta = 20; options.correction = true;
options

tds_gamma_r(ddae, 0, options)
