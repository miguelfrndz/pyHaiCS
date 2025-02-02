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
#define SPLIT_INTEGRAL

typedef struct {
    double omega;
    double t;
    double z;
    double kn;
    double kn_half;
} integration_params;

static double gsl_integrand_cos(double tau, void *params_void) {
    integration_params *params = (integration_params*) params_void;
    double omega    = params->omega;
    double t        = params->t;
    double z        = params->z;
    double kn       = params->kn;
    double kn_half  = params->kn_half;
    double tau_sq   = tau * tau;
    double u        = sqrt(fmax(tau_sq - z * z, 0.0));
    double cos_term = cos(omega * tau);
    
    if(u < ASYMPTOTIC_TOLERANCE)
        return cos_term * kn_half * ROUNDING_CORRECTION;
    else
        return cos_term * gsl_sf_bessel_J1(kn * u) / u * ROUNDING_CORRECTION;
}

static double gsl_integrand_sin(double tau, void *params_void) {
    integration_params *params = (integration_params*) params_void;
    double omega    = params->omega;
    double t        = params->t;
    double z        = params->z;
    double kn       = params->kn;
    double kn_half  = params->kn_half;
    double tau_sq   = tau * tau;
    double u        = sqrt(fmax(tau_sq - z * z, 0.0));
    double sin_term = sin(omega * tau);
    
    if(u < ASYMPTOTIC_TOLERANCE)
        return sin_term * kn_half * ROUNDING_CORRECTION;
    else
        return sin_term * gsl_sf_bessel_J1(kn * u) / u * ROUNDING_CORRECTION;
}

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

double integrand_sin(double tau, void* generic_params)
{
    const double* params = (double*)generic_params;
    const double k = params[0];
    const double t = params[1];
    const double z = params[2];
    const double omega = params[3];
    const double epsilon = params[4];

    const double u = sqrt(fmax(0.0, tau*tau - z*z)); // Precompute this part. The np.maximum avoids sqrt(-x) operations.

    if(u < epsilon)
        return sin(omega * tau) * k * 0.5 * 100.; // We use the Taylor expansion of J1(x) to avoid divisions by 0. 
    else
        return sin(omega * tau) * gsl_sf_bessel_J1(k * u) / u * 100.;
}

// We multiply the integrand by 100 and then divide the end result by 100 to mitigate roundoff errors
double integrand_cos(double tau, void* generic_params) {
    const double* params = (double*)generic_params;
    const double k = params[0];
    const double t = params[1];
    const double z = params[2];
    const double omega = params[3];
    const double epsilon = params[4];

    const double u = sqrt(fmax(0.0, tau*tau - z*z)); // Precompute this part. The np.maximum avoids sqrt(-x) operations.
    
    if(u < epsilon)
        return cos(omega * tau) * k * 0.5 * 100.; // We use the Taylor expansion of J1(x) to avoid divisions by 0. 
    else
        return cos(omega * tau) * gsl_sf_bessel_J1(k * u) / u * 100.;
}

void compute_integrals(double* partial_integral_cos, double* partial_integral_sin, double* x_min, double* x_max, 
                        double* k_n_values, double* t_values, double* z_values, 
                        int n_size, int t_size, int z_size, 
                        int limit, double epsabs, double epsrel, 
                        double omega, double epsilon) {
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int n = 0; n < n_size; ++n) {
        for (int t = 0; t < t_size; ++t) {
            if (t == t_size - 1) {
                printf("Running Iteration for n = %d...\n", (n + 1));
            }
            
            gsl_integration_workspace* workspace = gsl_integration_workspace_alloc(1024*1024*8);
            gsl_integration_cquad_workspace* workspace_cquad = gsl_integration_cquad_workspace_alloc(1024*1024*2);

            double err [[maybe_unused]];
            double params[5];

            gsl_function func_cos;
            func_cos.function = &integrand_cos;
            func_cos.params = &params;

            gsl_function func_sin;
            func_sin.function = &integrand_sin;
            func_sin.params = &params;

            params[0] = k_n_values[n];
            params[1] = t_values[t];
            params[3] = omega;
            params[4] = epsilon;

            for (int z = 0; z < z_size; ++z)
            {
                params[2] = z_values[z];

                gsl_integration_qag(&func_cos, x_min[t*z_size+z], x_max[t*z_size+z], 
                                 epsabs, epsrel, limit, 4, workspace, 
                                 &partial_integral_cos[(n*t_size+t)*z_size+z], &err);
                    
                gsl_integration_qag(&func_sin, x_min[t*z_size+z], x_max[t*z_size+z], 
                                 epsabs, epsrel, limit, 4, workspace, 
                                 &partial_integral_sin[(n*t_size+t)*z_size+z], &err);


                // We multiply the integrand by 100 and then divide the end result by 100 to mitigate roundoff errors
                partial_integral_cos[(n*t_size+t)*z_size+z] /= 100.;
                partial_integral_sin[(n*t_size+t)*z_size+z] /= 100.;

            }

            // We free the memory
            gsl_integration_workspace_free(workspace);
            gsl_integration_cquad_workspace_free(workspace_cquad);
        }
    }
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
        gsl_set_error_handler_off();
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
                    
                    #ifdef SPLIT_INTEGRAL
                        gsl_function F_cos;
                        F_cos.function = &gsl_integrand_cos;
                        F_cos.params   = &params;

                        gsl_function F_sin;
                        F_sin.function = &gsl_integrand_sin;
                        F_sin.params   = &params;

                        double result_cos, result_sin, error_cos, error_sin;
                        gsl_integration_workspace *workspace_qag_cos = gsl_integration_workspace_alloc(1e5);
                        gsl_integration_workspace *workspace_qag_sin = gsl_integration_workspace_alloc(1e5);
                        gsl_integration_cquad_workspace *workspace_cquad_cos = gsl_integration_cquad_workspace_alloc(1e5);
                        gsl_integration_cquad_workspace *workspace_cquad_sin = gsl_integration_cquad_workspace_alloc(1e5);

                        int status_qag_cos = gsl_integration_qag(&F_cos, min_x, max_x, EPS_ABS, EPS_REL, 
                                                LIMIT, GSL_INTEG_GAUSS41, 
                                                workspace_qag_cos, &result_cos, &error_cos);
                        int status_qag_sin = gsl_integration_qag(&F_sin, min_x, max_x, EPS_ABS, EPS_REL, 
                                                LIMIT, GSL_INTEG_GAUSS41, 
                                                workspace_qag_sin, &result_sin, &error_sin);

                        // If QAG fails, try CQUAD
                        if (status_qag_cos){
                            #ifdef DEBUG
                                printf("QAG for cos term failed. Trying CQUAD...\n");
                            #endif
                            gsl_integration_cquad(&F_cos, min_x, max_x, EPS_ABS, EPS_REL,
                                                workspace_cquad_cos, &result_cos, &error_cos, NULL);
                        }

                        if (status_qag_sin){
                            #ifdef DEBUG
                                printf("QAG for sin term failed. Trying CQUAD...\n");
                            #endif
                            gsl_integration_cquad(&F_sin, min_x, max_x, EPS_ABS, EPS_REL,
                                                workspace_cquad_sin, &result_sin, &error_sin, NULL);
                        }

                        double result = sin(omega * t) * (result_cos) - cos(omega * t) * (result_sin);

                        gsl_integration_workspace_free(workspace_qag_cos);
                        gsl_integration_workspace_free(workspace_qag_sin);
                        gsl_integration_cquad_workspace_free(workspace_cquad_cos);
                        gsl_integration_cquad_workspace_free(workspace_cquad_sin);
                    #else
                        gsl_function F;
                        F.function = &gsl_integrand;
                        F.params   = &params;
                        
                        double result, error;
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
                    #endif

                    integral[n * N_t * N_z + i * N_z + j] = result / ROUNDING_CORRECTION;
                #endif
            }
        }
    }
}
