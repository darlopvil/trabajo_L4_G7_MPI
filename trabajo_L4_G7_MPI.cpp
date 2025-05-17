#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <fstream>
#include <locale>
#include <iomanip>



// Estructura de resultados
struct ResultadoMontecarlo {
	double pi;
	double tiempo_segundos;
	double tiempo_ms;   
	double tiempo_us;   
	long long samples;
	int num_procesos;
	int rank;
	bool es_paralelo;
};


// Implementación secuencial
ResultadoMontecarlo montecarlo_secuencial(long long samples) {
	unsigned long long count = 0;

	double x, y;
	double inicio, final, total = 0;
	ResultadoMontecarlo resultado;

	resultado.samples = samples;
	resultado.es_paralelo = false;
	resultado.num_procesos = 1;


	inicio = (double)clock() / CLOCKS_PER_SEC;
	for (unsigned long long i = 0; i < (unsigned long long)samples; ++i)
	{
		//Crear un punto para tirar en el cuadrante
		x = ((double)rand()) / ((double)RAND_MAX);  //0 <= x <= 1
		y = ((double)rand()) / ((double)RAND_MAX);
		//Si el punto cae dentro del círculo, incrementar el contador
		if (x * x + y * y <= 1.0) {
			++count;
		}
	}
	final = (double)clock() / CLOCKS_PER_SEC;
	total = (final - inicio);

	// Guardar resultados
	resultado.pi = 4.0 * count / samples;
	resultado.tiempo_segundos = total;
	resultado.tiempo_ms = total * 1e3;
	resultado.tiempo_us = total * 1e6;

	printf("----------------MPI MonterCarlo Sin Paralelizar----------------\n");
	printf("Numero de Samples = %lld\n", samples);
	printf("pi = %.12f\n", resultado.pi);  //4 cuadrantes, solo se calcula en 1
	printf("Tiempo de ejec./elemento de calculo (en segundos) => %.12lf s\n", resultado.tiempo_segundos);
	printf("Tiempo de ejec./elemento de calculo (en milisegundos) => %.8lf ms\n", resultado.tiempo_ms);
	printf("Tiempo de ejec./elemento de calculo (en microsegundos) => %.8lf us\n", resultado.tiempo_us);
	printf("-------------------------------------------------------------------\n\n");

	return resultado;
}
// Implementación MPI
ResultadoMontecarlo montecarlo_mpi(long long samples) {
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	long long local_samples = samples / size;
	unsigned long long local_count = 0;
	double x, y;
	srand((unsigned int)time(NULL) + rank);// Semilla diferente por proceso

	double inicio = MPI_Wtime();

	for (long long i = 0; i < local_samples; ++i) {
		x = (double)rand() / RAND_MAX;
		y = (double)rand() / RAND_MAX;
		if (x * x + y * y <= 1.0) {
			++local_count;
		}
	}

	unsigned long long total_count = 0;
	MPI_Reduce(&local_count, &total_count, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

	double final = MPI_Wtime();

	ResultadoMontecarlo resultado;
	resultado.samples = samples;
	resultado.num_procesos = size;
	resultado.rank = rank;
	resultado.tiempo_segundos = final - inicio;
	resultado.tiempo_ms = resultado.tiempo_segundos * 1e3;
	resultado.tiempo_us = resultado.tiempo_segundos * 1e6;
	resultado.pi = (rank == 0) ? 4.0 * total_count / samples : 0.0;

	// Imprimir resultados solo desde el proceso 0
	if (rank == 0) {
		printf("----------------MPI MonterCarlo Paralelizado----------------\n");
		printf("Numero de Samples = %lld\n", samples);
		printf("Numero de Procesos = %d\n", size);
		printf("pi = %.12f\n", resultado.pi);
		printf("Tiempo de ejec./elemento de calculo (en segundos) => %.12lf s\n", resultado.tiempo_segundos);
		printf("Tiempo de ejec./elemento de calculo (en milisegundos) => %.8lf ms\n", resultado.tiempo_ms);
		printf("Tiempo de ejec./elemento de calculo (en microsegundos) => %.8lf us\n", resultado.tiempo_us);
	}

	return resultado;
}


void exportar_a_csv(const ResultadoMontecarlo& secuencial, const ResultadoMontecarlo& mpi) {
	// Usar append para añadir cada iteración al mismo archivo
	static bool primera_escritura = true;
	std::ofstream csv("resultados_montecarlo_mpi.csv", primera_escritura ? std::ios::trunc : std::ios::app);

	if (csv.is_open()) {
		// Escribir cabecera solo la primera vez
		if (primera_escritura) {
			csv << "Metodo;Samples;PI;Tiempo_s;Tiempo_ms;Tiempo_us;Num_Procesos\n";
			primera_escritura = false;
		}

		// Escribir enteros sin formato localizado
		csv << "Secuencial;" << secuencial.samples << ";";

		// Configurar para decimales con coma
		csv.imbue(std::locale("es_ES"));

		// Escribir valores con decimales
		csv << std::fixed << std::setprecision(12) << secuencial.pi << ";"
			<< std::fixed << std::setprecision(12) << secuencial.tiempo_segundos << ";"
			<< std::fixed << std::setprecision(8) << secuencial.tiempo_ms << ";"
			<< std::fixed << std::setprecision(8) << secuencial.tiempo_us << ";";

		// Volver a enteros
		csv.imbue(std::locale::classic());
		csv << secuencial.num_procesos << "\n";  // Eliminado el ";1" extra

		// MPI - misma estrategia
		csv << "MPI;" << mpi.samples << ";";

		// Configurar para decimales con coma
		csv.imbue(std::locale("es_ES"));

		csv << std::fixed << std::setprecision(12) << mpi.pi << ";"
			<< std::fixed << std::setprecision(12) << mpi.tiempo_segundos << ";"
			<< std::fixed << std::setprecision(12) << mpi.tiempo_ms << ";"
			<< std::fixed << std::setprecision(12) << mpi.tiempo_us << ";";

		// Volver a enteros
		csv.imbue(std::locale::classic());
		csv << mpi.num_procesos << "\n";  // Eliminado el ";1" extra

		csv.close();
	}
}






int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Array con los diferentes tamaños de muestras
	const long long sample_sizes[] = { 3000, 300000, 3000000 };
	const int num_iterations = 3;

	// Para cada tamaño de muestra, ejecutamos los cálculos
	for (int i = 0; i < num_iterations; i++) {
		long long samples = sample_sizes[i];

		if (rank == 0) {
			printf("\n\n============= ITERACION %d: SAMPLES = %lld =============\n\n", i + 1, samples);
		}

		ResultadoMontecarlo resultado_seq;
		if (rank == 0) {
			resultado_seq = montecarlo_secuencial(samples);
		}

		ResultadoMontecarlo resultado_mpi = montecarlo_mpi(samples);

		// Exportar resultados a CSV solo desde el proceso 0
		if (rank == 0) {
			exportar_a_csv(resultado_seq, resultado_mpi);
		}

		// Sincronización entre iteraciones para mantener la salida ordenada
		MPI_Barrier(MPI_COMM_WORLD);
	}

	MPI_Finalize();
	return 0;
}



