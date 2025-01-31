#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_sf_bessel.h>
#include <time.h>

static double asymptotic_j1(double x) {
    return sqrt(2.0 / (M_PI * x)) * cos(x - 3.0 * M_PI_4);
}

void monte_carlo_integrate(double *n_values, double *k_n_values, double *t_values, double *z_values, 
                           double *x_min, double *x_max, double omega, int N_max, int N_t, int N_z, 
                           int MC_samples, double *integral) {
    srand(time(NULL)); // Seed the random number generator

    for (int n = 0; n < N_max; n++) {
        double kn = k_n_values[n];
        printf("Running Iteration for n = %d...\n", (n + 1));

        for (int i = 0; i < N_t; i++) {
            double t = t_values[i];

            for (int j = 0; j < N_z; j++) {
                double z = z_values[j];
                double min_x = x_min[i * N_z + j];
                double max_x = x_max[i * N_z + j];

                if (max_x <= min_x) {
                    integral[n * N_t * N_z + i * N_z + j] = 0.0;
                    continue;
                }

                double sum = 0.0;
                #pragma omp parallel for reduction(+:sum)
                for (int s = 0; s < MC_samples; s++) {
                    double tau = min_x + (max_x - min_x) * ((double)rand() / RAND_MAX);
                    double u = sqrt(fmax(0, tau * tau - z * z));

                    #ifdef DEBUG
                        printf("n = %d, t = %d, z = %d, sample = %d, tau = %f, u = %f\n", n, i, j, s, tau, u);
                    #endif

                    double integrand_value;
                    integrand_value = sin(omega * (tau - t)) * kn * 0.5;
                    // if (u < 1e-6) {
                    //     integrand_value = sin(omega * (tau - t)) * kn * 0.5;
                    // } else {
                    //     // integrand_value = sin(omega * (tau - t)) * gsl_sf_bessel_J1(kn * u) / u;
                    //     integrand_value = sin(omega * (tau - t)) * asymptotic_j1(kn * u) / u;
                    // }
                    // double x = kn * u;
                    // if (x > 20.0) { // Threshold for asymptotic approximation
                    //     integrand_value = sin(omega * (tau - t)) * asymptotic_j1(x) / u;
                    // } else {
                    //     integrand_value = sin(omega * (tau - t)) * gsl_sf_bessel_J1(x) / u;
                    // }
                    #ifdef DEBUG
                        printf("integrand_value = %f, kn = %f, Bessel-J1 = %f\n", integrand_value, kn, gsl_sf_bessel_J1(kn * u));
                    #endif
                    sum += integrand_value;
                }

                integral[n * N_t * N_z + i * N_z + j] = (max_x - min_x) * (sum / MC_samples);
            }
        }
    }
}
