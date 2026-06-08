
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define DEFAULT_N  100000000LL
#define DEFAULT_K  3

int main(int argc, char *argv[]) {
    long long n;
    int k;           
    double *arr;
    double total_sum;
    double t_start, t_end;


    n = (argc > 1) ? atoll(argv[1]) : DEFAULT_N;
    k = (argc > 2) ? atoi(argv[2]) : DEFAULT_K;

#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    printf("=== OpenMP: Сума додатних елементів^%d ===\n", k);
    printf("Розмір масиву: %lld, кількість потоків: %d\n", n, nthreads);
#else
    printf("=== Послідовна версія (OpenMP не підключено) ===\n");
    printf("Розмір масиву: %lld\n", n);
#endif

    arr = (double *)malloc(n * sizeof(double));
    if (!arr) {
        fprintf(stderr, "Помилка виділення пам'яті для масиву розміром %lld\n", n);
        return 1;
    }

#pragma omp parallel
    {
#ifdef _OPENMP
        unsigned int seed = (unsigned int)(42 + omp_get_thread_num());
#else
        unsigned int seed = 42;
#endif
#pragma omp for schedule(static)
        for (long long i = 0; i < n; i++) {
            arr[i] = ((double)rand_r(&seed) / RAND_MAX) * 1000.0 - 500.0;
        }
    }


    total_sum = 0.0;

#ifdef _OPENMP
    t_start = omp_get_wtime();
#else
    clock_t c_start = clock();
#endif

#pragma omp parallel for schedule(static) reduction(+:total_sum)
    for (long long i = 0; i < n; i++) {
        if (arr[i] > 0.0) {
            total_sum += pow(arr[i], (double)k);
        }
    }

#ifdef _OPENMP
    t_end = omp_get_wtime();
    printf("Результат (сума): %.6e\n", total_sum);
    printf("Час виконання:    %.4f сек\n", t_end - t_start);
#else
    clock_t c_end = clock();
    printf("Результат (сума): %.6e\n", total_sum);
    printf("Час виконання:    %.4f сек\n",
           (double)(c_end - c_start) / CLOCKS_PER_SEC);
#endif

    free(arr);
    return 0;
}

