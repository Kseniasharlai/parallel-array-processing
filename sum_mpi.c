#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>


#define DEFAULT_N  100000000  
#define DEFAULT_K  3

int main(int argc, char *argv[]) {
    int rank, size;
    long long n;
    int k;
    double *arr = NULL;
    double *local_arr = NULL;
    long long local_n;
    double local_sum = 0.0;
    double global_sum = 0.0;
    double t_start, t_end;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    n = (argc > 1) ? atoll(argv[1]) : DEFAULT_N;
    k = (argc > 2) ? atoi(argv[2]) : DEFAULT_K;

    if (rank == 0) {
        printf("=== MPI: Сума додатних елементів^%d ===\n", k);
        printf("Розмір масиву: %lld, кількість процесів: %d\n", n, size);
    }

    local_n = n / size;

    long long remainder = n % size;

    if (rank == 0) {
        arr = (double *)malloc(n * sizeof(double));
        if (!arr) {
            fprintf(stderr, "Помилка виділення пам'яті\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        srand(42);
        for (long long i = 0; i < n; i++) {
        
            arr[i] = ((double)rand() / RAND_MAX) * 1000.0 - 500.0;
        }
    }

    local_arr = (double *)malloc(local_n * sizeof(double));
    if (!local_arr) {
        fprintf(stderr, "Процес %d: помилка виділення пам'яті\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }


    MPI_Barrier(MPI_COMM_WORLD);
    t_start = MPI_Wtime();

    MPI_Scatter(
        arr,
        (int)local_n,
        MPI_DOUBLE,
        local_arr,
        (int)local_n,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );


    local_sum = 0.0;
    for (long long i = 0; i < local_n; i++) {
        if (local_arr[i] > 0.0) {
            local_sum += pow(local_arr[i], (double)k);
        }
    }


    if (rank == 0 && remainder > 0) {
        for (long long i = local_n * size; i < n; i++) {
            if (arr[i] > 0.0) {
                local_sum += pow(arr[i], (double)k);
            }
        }
    }


    MPI_Reduce(
        &local_sum,
        &global_sum,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);
    t_end = MPI_Wtime();

    if (rank == 0) {
        printf("Результат (глобальна сума): %.6e\n", global_sum);
        printf("Час виконання: %.4f сек\n", t_end - t_start);
        free(arr);
    }

    free(local_arr);
    MPI_Finalize();
    return 0;
}
