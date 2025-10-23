#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    // Initialize MPI environment
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // current process ID
    MPI_Comm_size(MPI_COMM_WORLD, &size); // total number of processes

    // Default: 1,000,000 steps, can override via command line
    long long num_steps = 1000000LL;
    if (argc >= 2) {
        num_steps = atoll(argv[1]);
        if (num_steps <= 0) {
            if (rank == 0)
                fprintf(stderr, "Error: number of steps must be positive.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    double step = 1.0 / (double)num_steps;

    // Divides work evenly among processes (block distribution)
    long long base = num_steps / size;
    long long remainder = num_steps % size;
    long long start = rank * base + (rank < remainder ? rank : remainder);
    long long local_count = base + (rank < remainder ? 1 : 0);
    long long end = start + local_count;

    // Syncs all ranks before starting the timer
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // Local partial sum
    double local_sum = 0.0;
    for (long long i = start; i < end; i++) {
        double x = (i + 0.5) * step;
        local_sum += 4.0 / (1.0 + x * x);
    }

    // Combine results to process 0
    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    // Prints results from root process
    if (rank == 0) {
        double pi = step * global_sum;
        double elapsed = end_time - start_time;
        printf("PI is %.15f\n", pi);
        printf("Elapsed time = %.0f nanoseconds\n", elapsed * 1e9);
    }

    MPI_Finalize();
    return 0;
}
