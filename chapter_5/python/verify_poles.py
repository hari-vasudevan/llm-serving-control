import numpy as np, math

dt=1.0; tau1=2.0; tau2=3.0; B0=3; B_min=1; B_max=8

# FIFO plant: q[k+1] = q[k] + a - B
# e_q = q_ref - q_sw
# de_q[k+1] = de_q[k] + dB[k]   (A_aug=[1,0;1,1], B_aug=[1;0])
A = np.array([[1., 0.], [1., 1.]])
B = np.array([[1.], [0.]])

z1 = math.exp(-dt/tau1)
z2 = math.exp(-dt/tau2)
print(f"Desired poles: z1={z1:.4f}  z2={z2:.4f}")

C   = np.hstack([B, A@B])
e2  = np.array([[0., 1.]])
p_A = A@A - (z1+z2)*A + z1*z2*np.eye(2)
K   = (e2 @ np.linalg.inv(C) @ p_A).flatten()
print(f"Ackermann K: K_q={K[0]:.4f}  K_i={K[1]:.4f}")

A_cl = A - B @ K.reshape(1,2)
poles = np.linalg.eigvals(A_cl).real
print(f"CL poles: {[f'{p:.4f}' for p in poles]}  stable={all(abs(p)<1 for p in poles)}")

xi_min = (B0-B_max)/K[1]
xi_max = (B0-B_min)/K[1]
print(f"xi_q range: [{xi_min:.2f}, {xi_max:.2f}]")
print(f"K_q={K[0]:.4f}>0 so when q_sw>q_ref: e_q<0, dB=-(K_q*neg)>0, B increases (drains) OK={K[0]>0}")
