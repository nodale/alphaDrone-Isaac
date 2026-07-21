####################################
quick_v11a (custom 3D printed blade)
####################################

#from normalised thrust [0, 1] to thrust in kg
thrust_kg = f(T_n) = 0.96205*T_n^2 + 0.35267*T_n + -0.06114

#from thrust in newton to normalised thrust [0, 1]
T_n = g(thrust_N) = 0.16827*sqrt(thrust_N) + 0.02795*thrust_N + 0.08697

km = 0.021867

####################################
MS1302
####################################

#from normalised thrust [0, 1] to thrust in kg
thrust_kg = f(T_n) = 1.55377*T_n^2 + 0.25839*T_n + -0.04183

#from thrust in newton to normalised thrust [0, 1]
T_n = g(thrust_N) = 0.18288*sqrt(thrust_N) + 0.01155*thrust_N + 0.05060

km = 0.016125
