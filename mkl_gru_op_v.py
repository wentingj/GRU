
import theano
from theano import tensor, gof
from theano.tensor.blas import ldflags


class GRU(gof.Op):
    __props__ = ()

    def __init__(self, units=None, timesteps=None, batch_size=None, input_dim=None):
        self.units = units
	self.timesteps = timesteps
	self.batch_size = batch_size
	self.input_dim = input_dim
	super(GRU, self).__init__()

    #def make_node(self, in_):
    def make_node(self, x, x_m, h_init, W_h, W_x, b):
        x = tensor.as_tensor_variable(x)
        x_m = tensor.as_tensor_variable(x_m)
        h_init = tensor.as_tensor_variable(h_init)
        W_h = tensor.as_tensor_variable(W_h)
        W_x = tensor.as_tensor_variable(W_x)
        b = tensor.as_tensor_variable(b)
        out = [h_init.type()]
        return gof.Apply(self, [x, x_m, h_init, W_h, W_x, b], out)

    def c_headers(self):
        headers = ['<mkl.h>','<omp.h>']
        return headers

    def c_libraries(self):
        return ldflags()

    def c_support_code(self):
        ccode = """
        #define OMP_THRESHOLD (10)
        """
        return ccode

    def c_support_code_struct(self, node, name):
	#if node.inputs[0].type.dtype is 'float32':
        #    dtype = 'float'
        #elif node.inputs[0].type.dtype is 'float64':
        #    dtype = 'double'
        #else:
        #    raise TypeError('Gemm: dtype %s is not supported.'
        #                    % (node.inputs[0].type.dtype))
	dtype = 'float'

        timesteps = self.timesteps
        ccode = """
	    %(dtype)s** A;
            %(dtype)s** B;
            %(dtype)s** W_xmulx;

            MKL_INT    m_g[1];
            MKL_INT    k_g[1];
            MKL_INT    n_g[1];
            MKL_INT    lda_g[1];
            MKL_INT    ldb_g[1];
            MKL_INT    ldc_g[1];

            CBLAS_TRANSPOSE    transA_g[1];
            CBLAS_TRANSPOSE    transB_g[1];

            %(dtype)s  alpha_g[1];
            %(dtype)s  beta_g[1];
            MKL_INT    size_per_grp[1];
        """ % locals()
        return ccode

    def c_init_code_struct(self, node, name, sub):
        timesteps = self.timesteps
        ccode = """
            A = NULL;
            B = NULL;
            W_xmulx = NULL;

            m_g[0] = 0;
            k_g[0] = 0;
            n_g[0] = 0;

            lda_g[0] = 0;
            ldb_g[0] = 0;
            ldc_g[0] = 0;

            transA_g[0] = CblasNoTrans;
            transB_g[0] = CblasNoTrans;

            alpha_g[0] = 1.0;
            beta_g[0] = 1.0;
            size_per_grp[0] = 1;
	""" % locals()
        return ccode

    def c_cleanup_code_struct(self, node, name):
        ccode = """
            if (A) {
                free (A);
                A = NULL;
            }

            if (B) {
                free (B);
                B = NULL;
            }

            if (W_xmulx) {
                free (W_xmulx);
                W_xmulx =NULL;
            }
	"""

        return ccode

    def c_code(self, node, name, inputs, outputs, sub):
	units = self.units
        timesteps = self.timesteps
        batch_size = self.batch_size
        input_dim = self.input_dim
	x, x_m, h_init, W_h, W_x, b = inputs
	o, = outputs
	#
	#if node.inputs[0].type.dtype is 'float32':
        #    dtype = 's'
        #    d = 'float'
        #elif node.inputs[0].type.dtype is 'float64':
        #    dtype = 'd'
        #    d = 'double'
        #else:
        #    raise TypeError('Gemm: dtype %s is not supported.'
        #                    % (node.inputs[0].type.dtype))
        dtype = 's'
        d = 'float'
	
	print locals()
        ccode = """
	    int i,j;

            m_g[0] = %(units)s * 3;
            k_g[0] = %(input_dim)s;
            n_g[0] = %(batch_size)s;
            lda_g[0] = k_g[0];
            ldb_g[0] = n_g[0];
            ldc_g[0] = n_g[0];
            
            size_per_grp[0] = %(timesteps)s;

            if (A == NULL)
                A = (%(d)s**)malloc(%(timesteps)s * sizeof (%(d)s*));

            if (B == NULL)
                B = (%(d)s**)malloc(%(timesteps)s * sizeof (%(d)s*));

            if (W_xmulx == NULL)
                W_xmulx = (%(d)s**)malloc(%(timesteps)s * sizeof (%(d)s*));

            for (i = 0 ; i < %(timesteps)s; i ++) {
                A[i] = (%(d)s*) PyArray_DATA(%(W_x)s);
                B[i] = (%(d)s*) PyArray_DATA(%(x)s) + i * (%(batch_size)s) * (%(input_dim)s);
                W_xmulx[i] = (%(d)s*) PyArray_DATA(%(b)s) + i * (%(units)s*3) * (%(batch_size)s);
            }
            //xW+b
            cblas_%(dtype)sgemm_batch (
                        CblasRowMajor,
                        transA_g,
                        transB_g,
                        m_g,
                        n_g,
                        k_g,
                        alpha_g,
                        A,
                        lda_g,
                        B,
                        ldb_g,
                        beta_g,
                        W_xmulx,
                        ldc_g,
                        1,
                        size_per_grp);

            %(d)s alpha = 1.0;
	    int sz = %(units)s * %(batch_size)s;
            %(d)s *W_hmulh = (%(d)s*)mkl_malloc(sz * 3 * sizeof(%(d)s), 64 );
            %(d)s *zr_t = (%(d)s*)mkl_malloc(sz * 2 * sizeof(%(d)s), 64 );
            %(d)s *can_h_t = (%(d)s*)mkl_malloc(sz * sizeof(%(d)s), 64 );
            //%(d)s *h_tm1 = (%(d)s*)mkl_malloc(sz * sizeof(%(d)s), 64 );
            %(d)s *h_tm1 = NULL;

            npy_intp dims[2] = {0, 0}; 
	    dims[0] = PyArray_DIMS(%(h_init)s)[0];
            dims[1] = PyArray_DIMS(%(h_init)s)[1];
 
	    if (! %(o)s) {
                %(o)s = (PyArrayObject*) PyArray_ZEROS(2, dims, PyArray_TYPE(%(h_init)s), 0);
            }

	    h_tm1 = (%(d)s*)PyArray_DATA(%(o)s);
            for(i = 0; i < sz; i++){
                h_tm1[i] = ((%(d)s*) PyArray_DATA(%(h_init)s))[i];
            }
            for (i = 0; i < %(timesteps)s; i ++) {
		cblas_%(dtype)sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                //m, n, k, alpha, h_tm1, k, W_zh, n, beta, z_t, n);
                        %(units)s * 3, %(batch_size)s, %(units)s, alpha, (%(d)s*) PyArray_DATA(%(W_h)s), %(units)s, h_tm1, %(batch_size)s, 0.0, W_hmulh, %(batch_size)s);
		vsAdd(2 * sz, W_xmulx[i], W_hmulh, zr_t);
		vsExp(2 * sz, zr_t, zr_t);
		for(j = 0; j < sz*2; j++){
                    zr_t[j] = zr_t[j]/(zr_t[j] + 1);
                }
		vsMul(sz, W_hmulh + 2 * sz, zr_t + sz, can_h_t);
		vsAdd(sz, W_xmulx[i] + 2 * sz, can_h_t, can_h_t);
		vsTanh(sz, can_h_t, can_h_t);
		#pragma omp parallel for if(sz > OMP_THRESHOLD) private(j)
		for (j = 0; j < sz; j++){
                    //%(d)s h_tmp = (1 - zr_t[j]) * h_tm1[j] + zr_t[j] * can_h_t[j];
                    h_tm1[j] = (1 - zr_t[j]) * h_tm1[j] + zr_t[j] * can_h_t[j];
                    //h_tm1[j] = ((%(d)s*) PyArray_DATA(%(x_m)s))[i*%(units)s*%(batch_size)s+j] * h_tmp + (1 - ((%(d)s*)PyArray_DATA(%(x_m)s))[i*%(units)s*%(batch_size)s+j]) * h_tm1[j];
                }
            }
            mkl_free(can_h_t);
	
	""" % locals()
        return ccode

    def c_code_cache_version(self):
        return (1, 0, 0)
