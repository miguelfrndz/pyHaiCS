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

typedef struct {
    double omega;
    double t;
    double z;
    double kn;
    double kn_half;
} integration_params;

static double gsl_integrand(double tau, void *params_void) {
    integration_params *params = (integration_params*) params_void;
    double omega    = params->omega;
    double t        = params->t;
    double z        = params->z;
    double kn       = params->kn;
    double kn_half  = params->kn_half;
    double tau_sq   = tau * tau;
    double u        = sqrt(fmax(tau_sq - z * z, 0.0));
    double sin_term = (sin(omega * t) * cos(omega * tau) - cos(omega * t) * sin(omega * tau));
    
    if(u < ASYMPTOTIC_TOLERANCE)
        return sin_term * kn_half * ROUNDING_CORRECTION;
    else
        return sin_term * gsl_sf_bessel_J1(kn * u) / u * ROUNDING_CORRECTION;
}


static double asymptotic_j1(double x) {
    return sqrt(2.0 / (M_PI * x)) * cos(x - 3.0 * M_PI_4);
}

void monte_carlo_integrate(double *n_values, double *k_n_values, double *t_values, double *z_values, 
                           double *x_min, double *x_max, double omega, int N_max, int N_t, int N_z, 
                           int MC_samples, double *integral) {
    // srand(time(NULL)); // Seed the random number generator
    unsigned int base_seed = (unsigned int)time(NULL);
    #if SOLVING_MODE == SOLVING_MODE_MC
        printf(">>> Using Monte Carlo Integration.\n");
    #elif SOLVING_MODE == SOLVING_MODE_GSL
        printf(">>> Using GSL Integration.\n");
        // gsl_set_error_handler_off();
    #endif
    for (int n = 0; n < N_max; n++) {
        double kn = k_n_values[n];
        double kn_half = 0.5 * kn;
        printf("Running Iteration for n = %d...\n", (n + 1));
        
        // Loop over t and z indices in parallel
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < N_t; i++) {
            for (int j = 0; j < N_z; j++) {
                #ifdef DEBUG
                    printf("n = %d, t = %d, z = %d\n", n, i, j);
                #endif
                double t = t_values[i];
                double z = z_values[j];
                double z_sq = z * z;
                double min_x = x_min[i * N_z + j];
                double max_x = x_max[i * N_z + j];

                if (max_x <= min_x) {
                    integral[n * N_t * N_z + i * N_z + j] = 0.0;
                    continue;
                }

                #if SOLVING_MODE == SOLVING_MODE_MC
                    double sum = 0.0;
                    double range = max_x - min_x;
                    unsigned int thread_seed = base_seed + n * N_t * N_z + i * N_z + j + omp_get_thread_num();
                    #pragma omp parallel for reduction(+:sum)
                    for (int s = 0; s < MC_samples; s++) {
                        // double tau = min_x + range * ((double)rand() / RAND_MAX);
                        double tau = min_x + range * ((double)rand_r(&thread_seed) / RAND_MAX);
                        double tau_sq = tau * tau;
                        double u = sqrt(fmax(0, tau_sq - z_sq));

                        #ifdef DEBUG
                            printf("n = %d, t = %d, z = %d, sample = %d, tau = %f, u = %f\n", n, i, j, s, tau, u);
                        #endif

                        double integrand_value;
                        // double sin_term = sin(omega * (tau - t));
                        double sin_term = (sin(omega * t) * cos(omega * tau) - cos(omega * t) * sin(omega * tau));
                        if (u < ASYMPTOTIC_TOLERANCE) {
                            integrand_value = sin_term * kn_half * ROUNDING_CORRECTION;
                        } else {
                            integrand_value = sin_term * gsl_sf_bessel_J1(kn * u) / u * ROUNDING_CORRECTION;
                            // integrand_value = sin_term * asymptotic_j1(kn * u) / u * ROUNDING_CORRECTION;
                        }
                        #ifdef DEBUG
                            printf("integrand_value = %f, kn = %f, Bessel-J1 = %f\n", integrand_value, kn, gsl_sf_bessel_J1(kn * u));
                        #endif
                        sum += integrand_value / ROUNDING_CORRECTION;
                    }

                    integral[n * N_t * N_z + i * N_z + j] = range * (sum / MC_samples);
                #elif SOLVING_MODE == SOLVING_MODE_GSL
                    integration_params params;
                    params.omega    = omega;
                    params.t        = t;
                    params.z        = z;
                    params.kn       = kn;
                    params.kn_half  = kn_half;
                    
                    gsl_function F;
                    F.function = &gsl_integrand;
                    F.params   = &params;
                    
                    double result, error;
                    // gsl_integration_workspace *workspace_qag = gsl_integration_workspace_alloc(1024*1024*16);
                    // gsl_integration_cquad_workspace *workspace_cquad = gsl_integration_cquad_workspace_alloc(1024*1024*16);
                    gsl_integration_workspace *workspace_qag = gsl_integration_workspace_alloc(1e5);
                    gsl_integration_cquad_workspace *workspace_cquad = gsl_integration_cquad_workspace_alloc(1e5);
                    
                    int status_qag = gsl_integration_qag(&F, min_x, max_x, EPS_ABS, EPS_REL, 
                                            LIMIT, GSL_INTEG_GAUSS41, 
                                            workspace_qag, &result, &error);

                    // If QAG fails, try CQUAD
                    if (status_qag){
                        #ifdef DEBUG
                            printf("QAG failed. Trying CQUAD...\n");
                        #endif
                        gsl_integration_cquad(&F, min_x, max_x, EPS_ABS, EPS_REL,
                                            workspace_cquad, &result, &error, NULL);
                    }

                    gsl_integration_workspace_free(workspace_qag);
                    gsl_integration_cquad_workspace_free(workspace_cquad);

                    integral[n * N_t * N_z + i * N_z + j] = result / ROUNDING_CORRECTION;
                #endif
                }
        }
    }
}
