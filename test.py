import theano
from theano import tensor
import numpy as np

import mkl_gru_op_v

x = tensor.ftensor3('x')
x_m = tensor.ftensor3('x_m')
h_init = tensor.fmatrix('h_init')

W_h = tensor.fmatrix('W_h')
W_hzr = tensor.fmatrix('W_hzr')
W_hh = tensor.fmatrix('W_hh')
W_x = tensor.fmatrix('W_x')
b = tensor.ftensor3('b')

o = mkl_gru_op_v.GRU(units=1000, timesteps=10, batch_size=80, input_dim=620)(x, x_m, h_init, W_h, W_x, b)
f = theano.function([x, x_m, h_init, W_h, W_x, b], o)

units = 1000
timesteps = 10
batch_size = 80
input_dim = 620
x = np.random.rand(timesteps, input_dim, batch_size).astype(np.float32)
x_m = np.random.rand(timesteps, units, batch_size).astype(np.float32)-np.random.rand(timesteps, units, batch_size).astype(np.float32)
h_init = np.random.rand(units, batch_size).astype(np.float32)-np.random.rand(units, batch_size).astype(np.float32)
W_x = np.random.rand(units*3, input_dim).astype(np.float32)-np.random.rand(units*3, input_dim).astype(np.float32)
W_h = np.random.rand(units*3, units).astype(np.float32)-np.random.rand(units*3, units).astype(np.float32)
b = np.zeros((timesteps, units*3, batch_size), dtype=np.float32)-np.zeros((timesteps, units*3, batch_size), dtype=np.float32)

#####################################################################
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    return x
def GRU_NP():
    w_xz = W_x[:units,:] 
    w_xr = W_x[units:2*units,:]
    w_xh = W_x[2*units: ,:]
    w_hz = W_h[ :units,:] 
    w_hr = W_h[units:2*units,:]
    w_hh = W_h[2*units: ,:]
    b_z = b[:, :units,:]
    b_r = b[:, units:2*units,:]
    b_h = b[:, 2*units: ,:]
    hid = h_init
    for i in range(timesteps):
        x_z = np.dot(w_xz, x[i])
        x_r = np.dot(w_xr, x[i])
        x_h = np.dot(w_xh, x[i])
        
	t = x_z + np.dot(w_hz, hid) + b_z[i]
        z_t = sigmoid(t)

        t = x_r + np.dot(w_hr, hid) + b_r[i]
        r_t = sigmoid(t)

        t = x_h + r_t * np.dot(w_hh, hid) + b_h[i]
        can_h_t = np.tanh(t)
	
        h_t = (1. - z_t) * hid + z_t * can_h_t
        #hid = x_m[i] * h_t + (1. - x_m[i]) * hid
	hid = h_t
    return hid
   
o_numpy=GRU_NP()
print "numpy result="
print o_numpy
o = f(x, x_m, h_init, W_h, W_x, b)
print 'op result='
print o
assert np.allclose(o, o_numpy)
