#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_sf_bessel.h>
#include <omp.h>
#include <time.h>

static double asymptotic_j1(double x) {
    return sqrt(2.0 / (M_PI * x)) * cos(x - 3.0 * M_PI_4);
}

void monte_carlo_integrate(double *n_values, double *k_n_values, double *t_values, double *z_values, 
                           double *x_min, double *x_max, double omega, int N_max, int N_t, int N_z, 
                           int MC_samples, double *integral) {
    // srand(time(NULL)); // Seed the random number generator
    unsigned int base_seed = (unsigned int)time(NULL);
    for (int n = 0; n < N_max; n++) {
        double kn = k_n_values[n];
        double kn_half = 0.5 * kn;
        printf("Running Iteration for n = %d...\n", (n + 1));
        
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < N_t; i++) {
            for (int j = 0; j < N_z; j++) {
                double t = t_values[i];
                double z = z_values[j];
                double z_sq = z * z;
                double min_x = x_min[i * N_z + j];
                double max_x = x_max[i * N_z + j];

                if (max_x <= min_x) {
                    integral[n * N_t * N_z + i * N_z + j] = 0.0;
                    continue;
                }

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
                    double sin_term = (cos(omega * t) * sin(omega * tau) - sin(omega * t) * cos(omega * tau));
                    if (u < 1e-6) {
                        integrand_value = sin_term * kn_half;
                    } else {
                        integrand_value = sin_term * gsl_sf_bessel_J1(kn * u) / u;
                        // integrand_value = sin_term * asymptotic_j1(kn * u) / u;
                    }
                    #ifdef DEBUG
                        printf("integrand_value = %f, kn = %f, Bessel-J1 = %f\n", integrand_value, kn, gsl_sf_bessel_J1(kn * u));
                    #endif
                    sum += integrand_value;
                }

                integral[n * N_t * N_z + i * N_z + j] = range * (sum / MC_samples);
            }
        }
    }
}
