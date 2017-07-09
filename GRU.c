/* C source code is found in dgemm_example.c */

#define min(x,y) (((x) < (y)) ? (x) : (y))
#define    GRP_COUNT    2

#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include<sys/time.h>
#include <omp.h>

#define OMP_THRESHOLD 10


int loop = 100;

void  GRU_v(float *x, float *x_m, float* h_init,
	float* W_h, float* W_x,
	float *b,
	int units, int timesteps, int batch_size, int input_dim)
{
    int i,j;
    float** A;
    float** B;
    float** W_xmulx;
    A = NULL;
    B = NULL;
    W_xmulx = NULL;
    MKL_INT    m_g[1] = {units*3};
    MKL_INT    k_g[1] = {input_dim};
    MKL_INT    n_g[1] = {batch_size};
    MKL_INT    lda_g[1] = {input_dim/*k*/};
    MKL_INT    ldb_g[1] = {batch_size/*n*/};
    MKL_INT    ldc_g[1] = {batch_size/*n*/};
    CBLAS_TRANSPOSE    transA_g[1] = {CblasNoTrans};
    CBLAS_TRANSPOSE    transB_g[1] = {CblasNoTrans};
    float    alpha_g[1] = {1.0};
    float    beta_g[1] = {1.0};
    MKL_INT    size_per_group[1] = {timesteps};
    if (A == NULL)
        A = (float**)malloc(timesteps * sizeof (float*));

    if (B == NULL)
        B = (float**)malloc(timesteps * sizeof (float*));

    if (W_xmulx == NULL)
        W_xmulx = (float**)malloc(timesteps * sizeof (float*));

    for (i = 0 ; i < timesteps; i ++) {
        A[i] = W_x;
        B[i] = x + i * input_dim*batch_size;
        W_xmulx[i] = b + i * units*3* batch_size;
    }
    cblas_sgemm_batch (
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
                size_per_group);

    float alpha = 1.0;
    int sz = units * batch_size;
    float *h_tm1 = (float*)mkl_malloc(sz * sizeof(float), 64 );
    float *W_hmulh = (float*)mkl_malloc(3 * sz * sizeof(float), 64 );
    float *zr_t = (float*)mkl_malloc(2 * sz * sizeof(float), 64);
    float *can_h_t = (float*)mkl_malloc(sz * sizeof(float), 64);
    for(i=0; i<batch_size*units; i++){
        h_tm1[i] = h_init[i];
    }
    //step
    //z_t = K.sigmoid(x_z + K.dot(h_tm1, self.W_hz) + self.b_z)
    //r_t = K.sigmoid(x_r + K.dot(h_tm1, self.W_hr) + self.b_r)
    //can_h_t = K.tanh(x_h + r_t * K.dot(h_tm1, self.W_hh) + self.b_h)
    //h_t = (1. - z_t) * h_tm1 + z_t * can_h_t
    //h_t = x_m * h_t + (1. - x_m) * h_tm1
    for(i=0; i < timesteps; i++){
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                //m, n, k, alpha, h_tm1, k, W_zh, n, beta, z_t, n);
                units * 3, batch_size, units, alpha, W_h, units, h_tm1, batch_size, 0.0, W_hmulh, batch_size);
    	vsAdd( 2*sz, W_xmulx[i], W_hmulh, zr_t );
	vsExp( 2 * sz, zr_t, zr_t);
	vsMul( sz, W_hmulh + 2*sz, zr_t + sz, can_h_t );
	vsAdd( sz, W_xmulx[i] + 2*sz, can_h_t, can_h_t);
	vsTanh( sz, can_h_t, can_h_t );
	int sz = units*batch_size;
	#pragma omp parallel for if(sz > OMP_THRESHOLD) private(j)
	for(j=0; j<sz;j++){
	    //float h_tmp = (1 - zr_t[j])*h_tm1[j] + zr_t[j]*can_h_t[j];
	    h_tm1[j] = (1 - zr_t[j])*h_tm1[j] + zr_t[j]*can_h_t[j];
	    //h_tm1[j] = x_m[j]*h_tmp + (1 - x_m[j])*h_tm1[j];
	    fprintf(stderr, "%f  ", h_tm1[j]);
	}
    }
    free(A);
    free(B);
    free(W_xmulx);
    mkl_free(h_tm1);
    mkl_free(W_hmulh);
    mkl_free(zr_t);
    mkl_free(can_h_t);
}

int main(int argc, char** argv)
{
    int i,j;
    int units, timesteps, batch_size, input_dim;
    float *x;
    float *x_m;
    float *h_init;
    float *W_h, *W_x;
    float *b_h, *b_v;

    units = 4;
    timesteps = 1;
    batch_size = 2;
    input_dim = 3;
    fprintf(stderr,"units = %d\n",units);
    fprintf(stderr,"timesteps = %d\n",timesteps);
    fprintf(stderr,"batch_size = %d\n",batch_size);
    fprintf(stderr,"input_dim = %d\n",input_dim);
    
    x = (float *)mkl_malloc( timesteps*batch_size*input_dim*sizeof( float ), 64 );
    x_m = (float *)mkl_malloc( timesteps*batch_size*units*sizeof( float ), 64 );
    h_init = (float *)mkl_malloc( batch_size*units*sizeof( float ), 64 );
    W_h = (float *)mkl_malloc( 3*units*units*sizeof( float ), 64 );
    W_x = (float *)mkl_malloc( input_dim*units*3*sizeof( float ), 64 );

    b_v = (float *)mkl_malloc( timesteps*batch_size*units*3*sizeof( float ), 64 );
    
    for (i = 0; i < (timesteps*batch_size*input_dim); i++) {
	x[i] = (float)((i+1.0)/100.0);
	x_m[i] = 1.0;
    }

    for (i = 0; i < (batch_size*units); i++) {
        h_init[i] = (float)((i+2.0)/100.0);
    }

    for (i = 0; i < (units*units*3); i++) {
	W_h[i] = (float)((i)/100.0);
    }

    for (i = 0; i < (input_dim*units*3); i++) {
        W_x[i] = (float)((i-1)/50.0);
    }

    for (i = 0; i < (timesteps*batch_size*units*3); i++) {
        b_v[i] = 0.0;
    }

    struct timeval start_v;
    struct timeval end_v;
    gettimeofday(&start_v,NULL);
    GRU_v(x, x_m, h_init, W_h, W_x, b_v, units, timesteps, batch_size, input_dim);
    gettimeofday(&end_v,NULL);
    
    float duration_v = (end_v.tv_sec - start_v.tv_sec) * 1000 + (float)(end_v.tv_usec - start_v.tv_usec) /1000;
    fprintf(stderr,"        time_v = %.4f ms\n\n",duration_v/loop );
    
    mkl_free(x);
    mkl_free(x_m);
    mkl_free(h_init);
    mkl_free(W_h);
    mkl_free(W_x);
    mkl_free(b_v);
    return 0;
} 
