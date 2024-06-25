ndde = tds_create_neutral({-3/4,1/2},[1 2],{1/4,-1/3},[0 1]);
cD = -log(sqrt(2));
options=tds_roots_options('max_size_evp',1500);
cr = tds_roots(ndde,[-0.9 0.2 -500 500],options);
tds_eigenplot(cr,'Markersize',4);
hold on
plot([cD cD],ylim,'b--')
xlim([-0.9 0.2])
ylim([-500,500])