clear all;
close all;

%import tds_control.stability.compute_N_rhp
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

r = -2.5

%%
rdde = tds_create(A, hA);
rdde.E

roots = tds_roots(rdde, r);

%%

mA=length(hA); % number of delay terms
% scale tds such that maximal delay-value equals 1
% lambda_hat = lambda*tau_m
% det(lambda E - A0 - A1 *exp(-lambda tau_1) - ... - Am *exp(-lambda tau_m)) = 0
% => det(lambda_hat E - tau_m *A0 - tau_m * A1 *exp(-lambda_hat tau_1/tau_m) - ... - tau_m Am *exp(-lambda_hat)) = 0
% re-scaled system matrices and delays are stored in K and tau_s
tau_s=hA/hA(mA);
K = cell(1,length(A));
for i=1:length(A)
	K{i}=hA(mA)*A{i};
end

% case rhp
rs=r*hA(mA);
% introduce a shift of the origin, shifted matrices are stored in B and C
B=K{1}+(-rs)*E;
C=zeros(n,n,mA-1);
for ii=1:mA-1
	C(:,:,ii)=K{ii+1}*exp((-rs)*tau_s(ii+1));
end

%%
compute_N_rhp(E,B,C,tau_s)
