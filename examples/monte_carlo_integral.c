#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>
#include <omp.h>
#include <time.h>

#define ASYMPTOTIC_TOLERANCE 1e-6
#define ROUNDING_CORRECTION 1e2
#define EPS_ABS 1e-7
#define EPS_REL 1e-4
#define LIMIT 100000
#define SOLVING_MODE_MC 1
#define SOLVING_MODE_GSL 2
#define SOLVING_MODE SOLVING_MODE_GSL
#define GSL_ALLOC_SIZE 1024*1024*8

typedef struct {
    double omega;
    double t;
    double z;
    double kn;
    double kn_half;
} integration_params;

static double asymptotic_j1(double x) {
    return sqrt(2.0 / (M_PI * x)) * cos(x - 3.0 * M_PI_4);
}

static inline double integrand_sin(double tau, void* params_void) {
    integration_params *params = (integration_params*) params_void;
    double omega    = params->omega;
    double t        = params->t;
    double z        = params->z;
    double kn       = params->kn;
    double kn_half  = params->kn_half;
    double u        = sqrt(fmax(tau * tau - z * z, 0.0));
    double sin_term = sin(omega * tau);

    if(u < ASYMPTOTIC_TOLERANCE)
        // We use the Taylor expansion of J1(x) to avoid divisions by 0.
        return sin_term * kn_half * ROUNDING_CORRECTION; 
    else
        return sin_term * gsl_sf_bessel_J1(kn * u) / u * ROUNDING_CORRECTION;
}

static inline double integrand_cos(double tau, void* params_void) {
    integration_params *params = (integration_params*) params_void;
    double omega    = params->omega;
    double t        = params->t;
    double z        = params->z;
    double kn       = params->kn;
    double kn_half  = params->kn_half;
    double u        = sqrt(fmax(tau * tau - z * z, 0.0));
    double cos_term = cos(omega * tau);
    
    if(u < ASYMPTOTIC_TOLERANCE)
        return cos_term * kn_half * ROUNDING_CORRECTION;
    else
        return cos_term * gsl_sf_bessel_J1(kn * u) / u * ROUNDING_CORRECTION;
}

void compute_integrals(double* partial_integral_cos, double* partial_integral_sin, double* x_min, 
                        double* x_max, double* k_n_values, double* t_values, double* z_values, 
                        int n_size, int t_size, int z_size, double omega, int MC_SAMPLES) {
    // srand(time(NULL)); // Seed the random number generator
    unsigned int base_seed = (unsigned int)time(NULL);
    #if SOLVING_MODE == SOLVING_MODE_MC
        printf(">>> Using Monte Carlo Integration (w/ %d MC-Samples).\n", MC_SAMPLES);
    #elif SOLVING_MODE == SOLVING_MODE_GSL
        printf(">>> Using GSL Integration.\n");
        gsl_set_error_handler_off();
    #endif
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int n = 0; n < n_size; ++n) {
        for (int i = 0; i < t_size; ++i) {
            if (i == 0) {
                printf("Running Iteration for n = %d...\n", (n + 1));
            }

            double kn = k_n_values[n];
            double kn_half = 0.5 * kn;
            double t = t_values[i];

            integration_params params;
            params.omega = omega;
            params.t = t;
            params.kn = kn;
            params.kn_half = kn_half;
            
            #if SOLVING_MODE == SOLVING_MODE_MC
                for (int j = 0; j < z_size; ++j) {
                    params.z = z_values[j];
                    double min = x_min[i*z_size+j];
                    double max = x_max[i*z_size+j];
                    double range = max - min;
                    int _index = (n*t_size+i)*z_size+j;
                    unsigned int thread_seed = base_seed + _index + omp_get_thread_num();

                    double sum_cos = 0.0;
                    double sum_sin = 0.0;

                    #pragma omp parallel for reduction(+:sum_cos, sum_sin) schedule(dynamic)
                    for (int k = 0; k < MC_SAMPLES; ++k) {
                        double tau = min + range * ((double)rand_r(&thread_seed) / RAND_MAX);
                        sum_cos += integrand_cos(tau, &params);
                        sum_sin += integrand_sin(tau, &params);
                    }

                    partial_integral_cos[_index] = range * sum_cos / MC_SAMPLES;
                    partial_integral_sin[_index] = range * sum_sin / MC_SAMPLES;

                    partial_integral_cos[_index] /= ROUNDING_CORRECTION;
                    partial_integral_sin[_index] /= ROUNDING_CORRECTION;
                }
            #elif SOLVING_MODE == SOLVING_MODE_GSL
                gsl_integration_workspace* workspace = gsl_integration_workspace_alloc(GSL_ALLOC_SIZE);
                gsl_integration_cquad_workspace* workspace_cquad = gsl_integration_cquad_workspace_alloc(GSL_ALLOC_SIZE);

                gsl_function func_cos;
                func_cos.function = &integrand_cos;
                func_cos.params = &params;

                gsl_function func_sin;
                func_sin.function = &integrand_sin;
                func_sin.params = &params;

                double err;
                
                for (int j = 0; j < z_size; ++j) {
                    params.z = z_values[j];
                    double min = x_min[i*z_size+j];
                    double max = x_max[i*z_size+j];
                    int _index = (n*t_size+i)*z_size+j;

                    int status_qag_cos = gsl_integration_qag(&func_cos, min, max, 
                                            EPS_ABS, EPS_REL, LIMIT, GSL_INTEG_GAUSS41, workspace, 
                                            &partial_integral_cos[_index], &err);
                        
                    int status_qag_sin = gsl_integration_qag(&func_sin, min, max, 
                                            EPS_ABS, EPS_REL, LIMIT, GSL_INTEG_GAUSS41, workspace, 
                                            &partial_integral_sin[_index], &err);
                    
                    // If QAG fails, try CQUAD
                        if (status_qag_cos){
                            #ifdef DEBUG
                                printf("QAG for cos term failed. Trying CQUAD...\n");
                            #endif
                            gsl_integration_cquad(&func_cos, min, max, EPS_ABS, EPS_REL,
                                                workspace_cquad, &partial_integral_cos[_index], &err, NULL);
                        }

                        if (status_qag_sin){
                            #ifdef DEBUG
                                printf("QAG for sin term failed. Trying CQUAD...\n");
                            #endif
                            gsl_integration_cquad(&func_sin, min, max, EPS_ABS, EPS_REL,
                                                workspace_cquad, &partial_integral_sin[_index], &err, NULL);
                        }

                    // We multiply the integrand by C and then divide the end result by C to mitigate roundoff errors
                    partial_integral_cos[_index] /= ROUNDING_CORRECTION;
                    partial_integral_sin[_index] /= ROUNDING_CORRECTION;

                }

                // We free the memory
                gsl_integration_workspace_free(workspace);
                gsl_integration_cquad_workspace_free(workspace_cquad);
            #endif
        }
    }
}