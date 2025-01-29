clear all;
close all;

import tds_control.stability.compute_gamma_r
DD1 =[
    [0,0,0,-3]
    [0,0,0,-4]
    [0,0,0,-1]
    [0,0,0,0]
];
    
DD2 = [
    [0,0,0,-0.4]
    [0,0,0,0.4]
    [0,0,0,0.4]
    [0,0,0,0]
];
DD3 = [
    [0,0,0,0]
    [0,0,0,0]
    [0,0,0,0]
    [-0.01,-0.01,-0.01,0]
];

hDD = [2.5, 5, 0];

options.correction=true;
[gammar, out] = compute_gamma_r({DD1, DD2, DD3}, hDD, 0, options)
