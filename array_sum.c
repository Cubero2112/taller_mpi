#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

//#define VECTOR_SIZE 100
//#define VECTOR_SIZE 1000000
#define VECTOR_SIZE 10000000

int main(int argc, char** argv) {
	int rank, size;
	double start_time, end_time;
	int namelen;
	char processor_name[MPI_MAX_PROCESSOR_NAME];

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Get_processor_name(processor_name, &namelen);
	
	if (rank == 0) printf("Size %d \n", size);
	printf("Current rank %d \n", rank);

	fprintf(stdout, "Process %d of %d is on %s\n", rank, size, processor_name);
	fflush(stdout);	

	// Tamaño del vector local para cada proceso
	int local_size = VECTOR_SIZE / size;
	if (rank == 0) printf("LOcal size for each process: %d \n", local_size);

	// Inicialización de vectores locales y globales
	int* local_vector_a = (int*)malloc(local_size * sizeof(int));
	int* local_vector_b = (int*)malloc(local_size * sizeof(int));
	int* local_result = (int*)malloc(local_size * sizeof(int));

	int* global_vector_a = NULL;
	int* global_vector_b = NULL;
	int* global_result = NULL;

	if (rank == 0) {
    	global_vector_a = (int*)malloc(VECTOR_SIZE * sizeof(int));
    	global_vector_b = (int*)malloc(VECTOR_SIZE * sizeof(int));
    	global_result = (int*)malloc(VECTOR_SIZE * sizeof(int));

    	// Inicialización de vectores globales
    	for (int i = 0; i < VECTOR_SIZE; ++i) {
        	global_vector_a[i] = i + 1;
        	global_vector_b[i] = VECTOR_SIZE - i;
    	}
	}

	MPI_Barrier(MPI_COMM_WORLD); // Sincronizar todos los procesos antes de medir el tiempo

	// Iniciar el temporizador
	start_time = MPI_Wtime();

	// Distribución de partes de vectores globales a los procesos locales
	MPI_Scatter(global_vector_a, local_size, MPI_INT, local_vector_a, local_size, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Scatter(global_vector_b, local_size, MPI_INT, local_vector_b, local_size, MPI_INT, 0, MPI_COMM_WORLD);

	// Suma local de vectores
	for (int i = 0; i < local_size; ++i) {
    	local_result[i] = local_vector_a[i] + local_vector_b[i];
	}

	// Recopilación de resultados locales en el vector global
	MPI_Gather(local_result, local_size, MPI_INT, global_result, local_size, MPI_INT, 0, MPI_COMM_WORLD);

	// Detener el temporizador
	end_time = MPI_Wtime();

	// Impresión de resultados y tiempo en el proceso 0
	if (rank == 0) {
	/*	
    	printf("Vector A: ");
    	for (int i = 0; i < VECTOR_SIZE; ++i) {
        	printf("%d ", global_vector_a[i]);
    	}
    	printf("\n");

    	printf("Vector B: ");
    	for (int i = 0; i < VECTOR_SIZE; ++i) {
        	printf("%d ", global_vector_b[i]);
    	}
    	printf("\n");

    	printf("Resultado: ");
    	for (int i = 0; i < VECTOR_SIZE; ++i) {
        	printf("%d ", global_result[i]);
    	}
    	printf("\n");
	*/
    	// Imprimir el tiempo transcurrido
    	printf("Tiempo de ejecución: %f segundos\n", end_time - start_time);
	}

	// Liberar memoria
	free(local_vector_a);
	free(local_vector_b);
	free(local_result);
	if (rank == 0) {
    	free(global_vector_a);
    	free(global_vector_b);
    	free(global_result);
	}

	MPI_Finalize();
	return 0;
}

