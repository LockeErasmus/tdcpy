clear all;
close all;

ndde=tds_create_neutral({-3/4,1/2},[1 2],{1/4,-1/3},[0 1]);
ddae=ndde.to_ddae()

%ddae.A

%ddae.get_delay_difference_equation()

ddae2 = tds_create_ddae(ddae.E, ddae.A, ddae.hA)