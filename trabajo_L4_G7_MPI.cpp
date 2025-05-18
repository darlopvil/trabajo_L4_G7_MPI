/*
 ============================================================================
 SOBRE LA GENERACIÓN DE NÚMEROS ALEATORIOS EN MPI:
 ----------------------------------------------------------------------------
 Esta implementación utiliza un enfoque optimizado para la generación de números
 aleatorios mediante MPI:

 1. GENERADORES DE NÚMEROS ALEATORIOS:
	- Se emplea el algoritmo Mersenne Twister (MT19937), un generador de alta calidad
	  con periodo 2^19937-1, capaz de producir 623 dimensiones de equidistribución.
	- Este generador supera ampliamente a los generadores lineales congruenciales
	  tradicionales en términos de calidad estadística y distribución uniforme.
	- Comparación con otras alternativas:
	  * Superior a rand() del estándar C: mejor distribución y mayor periodo
	  * Superior a minstd_rand: que solo tiene periodo 2^31-1

 2. ESTRATEGIA DE SEMILLAS:
	- Esta implementación combina múltiples fuentes de entropía para cada proceso:
	  a) time(NULL): segundos desde la época Unix (baja entropía entre procesos simultáneos)
	  b) clock(): tiempo de CPU de alta precisión (varía incluso entre procesos sincronizados)
	  c) random_device: entropía del hardware cuando está disponible
	  d) ID del proceso (rank) multiplicado por un número primo grande (7919)
	  e) Operación XOR (^) para combinar estas fuentes preservando la entropía de bits

	- La fórmula: seed = time(NULL) ^ (clock()*113) ^ random_device() ^ (rank*7919)
	  asegura semillas únicas incluso cuando todos los procesos inician simultáneamente
	  en el mismo segundo.

 3. DISTRIBUCIÓN UNIFORME:
	- Se utiliza std::uniform_real_distribution<double>(0.0, 1.0) para transformar
	  los valores generados a una distribución perfectamente uniforme en [0,1).
	- Esta distribución garantiza que cada valor del intervalo tiene exactamente
	  la misma probabilidad de aparecer, eliminando sesgos inherentes a métodos más
	  simples como la división por RAND_MAX.

 ============================================================================
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <fstream>
#include <locale>
#include <iomanip>
#include <random>	// Para el generador de números aleatorios (C++11)
// Hacer visibles las constantes matemáticas estándar como M_PI
#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>



// Estructura para almacenar los resultados de cada experimento
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


// Implementación secuencial del método de Monte Carlo para PI
ResultadoMontecarlo montecarlo_secuencial(long long samples) {
	unsigned long long count = 0;

	double x, y;
	double inicio, final, total = 0;
	ResultadoMontecarlo resultado;

	resultado.samples = samples;
	resultado.es_paralelo = false;
	resultado.num_procesos = 1;

	// GENERACIÓN DE SEMILLA DE ALTA CALIDAD PARA VERSIÓN SECUENCIAL
	// ----------------------------------------------------------------------
	// Se combinan múltiples fuentes de entropía para crear una semilla única:
	// 1. random_device: Acceso a generador de entropía del hardware/OS
	std::random_device rd;

	// 2. Combinando fuentes mediante XOR (mejor que suma o multiplicación):
	//    - time(NULL): segundos desde 1970 (cambia lentamente)
	//    - clock(): pulsos de reloj desde inicio del programa (alta precisión)
	//    - rd(): valor aleatorio del generador hardware (máxima entropía)
	//    - El primo 113 ayuda a distribuir bits de clock() en la semilla
	unsigned int seed = static_cast<unsigned int>(time(NULL)) ^ rd() ^ (clock() * 113);

	// Mersenne Twister de 32 bits (MT19937)
	// - Periodo ultralargo: puede generar 2^19937-1 números antes de repetir ciclo
	// - Equidistribución en 623 dimensiones (ideal para simulaciones Monte Carlo)
	std::mt19937 gen(seed);

	// Transformación a distribución continua uniforme [0,1)
	// Garantiza distribución perfectamente uniforme sin sesgos
	std::uniform_real_distribution<double> dis(0.0, 1.0);

	// Medición de tiempo de ejecución
	inicio = (double)clock() / CLOCKS_PER_SEC;
	for (unsigned long long i = 0; i < (unsigned long long)samples; ++i)
	{
		// Generación de coordenadas aleatorias en el cuadrante [0,1)×[0,1)
		x = dis(gen);  // Coordenada X
		y = dis(gen);  // Coordenada Y

		// Comprobación si el punto (x,y) cae dentro del círculo unitario
		// Un punto está dentro del círculo si x²+y² ≤ 1
		if (x * x + y * y <= 1.0) {
			++count;  // Incrementamos contador de puntos dentro del círculo
		}
	}
	final = (double)clock() / CLOCKS_PER_SEC;
	total = (final - inicio);

	// Cálculo de PI basado en la proporción de puntos dentro del círculo
	// π = 4 × (puntos_dentro / total_puntos)
	resultado.pi = 4.0 * count / samples;
	resultado.tiempo_segundos = total;
	resultado.tiempo_ms = total * 1e3;
	resultado.tiempo_us = total * 1e6;

	// Impresión de resultados por consola
	printf("----------------MPI MonterCarlo Sin Paralelizar----------------\n");
	printf("Numero de Samples = %lld\n", samples);
	printf("pi = %.12f\n", resultado.pi);  // 4 cuadrantes, solo se calcula en 1
	printf("Tiempo de ejec./elemento de calculo (en segundos) => %.12lf s\n", resultado.tiempo_segundos);
	printf("Tiempo de ejec./elemento de calculo (en milisegundos) => %.8lf ms\n", resultado.tiempo_ms);
	printf("Tiempo de ejec./elemento de calculo (en microsegundos) => %.8lf us\n", resultado.tiempo_us);
	printf("-------------------------------------------------------------------\n\n");

	printf("Error absoluto: %.12f (%.8f%%)\n",
		fabs(resultado.pi - M_PI),
		100.0 * fabs(resultado.pi - M_PI) / M_PI);

	return resultado;
}

// Implementación paralela con MPI del método de Monte Carlo para PI
ResultadoMontecarlo montecarlo_mpi(long long samples) {
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// DISTRIBUCIÓN EQUITATIVA DEL TRABAJO
	// ----------------------------------------------------------------------
	// División base: dividir el total de muestras entre procesos
	long long local_samples = samples / size;

	// Manejo de residuo: si la división no es exacta, distribuir el resto
	// Asignamos una muestra extra a los primeros 'samples % size' procesos
	// Esto garantiza que la suma de local_samples de todos los procesos
	// sea exactamente igual a 'samples' (importante para la precisión)
	if (rank < samples % size) {
		local_samples++;
	}

	unsigned long long local_count = 0;
	double x, y;

	// GENERACIÓN DE SEMILLAS ÚNICAS PARA CADA PROCESO
	// ----------------------------------------------------------------------
	// Problema crítico: asegurar secuencias aleatorias independientes entre procesos
	// que pueden iniciar simultáneamente en el mismo segundo

	// 1. Hardware entropy source (si está disponible)
	std::random_device rd;

	// 2. Estrategia de múltiples fuentes combinadas mediante XOR:
	//    a) time(NULL): segundos desde epoca Unix (baja variación entre procesos)
	//    b) clock()*113: tiempo CPU de precisión escalado por un primo
	//    c) rd(): entropía pura del hardware (máxima calidad)
	//    d) rank*7919: perturbación única por proceso
	//       - Se usa 7919 por ser primo grande, reduciendo colisiones
	//       - La multiplicación distribuye el efecto del rank en múltiples bits
	//
	// Esta estrategia garantiza semillas diferentes incluso si:
	// - Todos los procesos inician exactamente en el mismo segundo
	// - Los valores de clock() son similares entre procesos
	// - La implementación de random_device es determinista
	unsigned int seed = static_cast<unsigned int>(time(NULL)) ^
		(static_cast<unsigned int>(clock() * 113)) ^
		rd() ^
		(rank * 7919);

	// Generador Mersenne Twister (MT19937): estado interno de 19937 bits
	// Ideal para simulaciones Monte Carlo por su excelente distribución multidimensional
	std::mt19937 rng(seed);

	// Distribución uniforme en rango [0,1) - asegura perfecta equidistribución
	std::uniform_real_distribution<double> dist(0.0, 1.0);

	// Medición de tiempo de ejecución con MPI_Wtime (reloj de alta precisión de MPI)
	double inicio = MPI_Wtime();

	// CÁLCULO LOCAL EN CADA PROCESO
	// ----------------------------------------------------------------------
	// Cada proceso genera y evalúa su conjunto asignado de puntos aleatorios
	for (long long i = 0; i < local_samples; ++i) {
		x = dist(rng);  // Coordenada X aleatoria
		y = dist(rng);  // Coordenada Y aleatoria

		// Verificar si el punto (x,y) está dentro del círculo unitario
		if (x * x + y * y <= 1.0) {
			++local_count;  // Incrementar contador local
		}
	}

	// REDUCCIÓN GLOBAL: COMBINAR RESULTADOS DE TODOS LOS PROCESOS
	// ----------------------------------------------------------------------
	// MPI_Reduce realiza una operación de reducción (suma) sobre valores de todos los procesos
	// - Parámetros:
	//   * &local_count: dirección del valor local a reducir
	//   * &total_count: dirección donde se almacenará el resultado (solo en proceso root)
	//   * 1: número de elementos a reducir
	//   * MPI_UNSIGNED_LONG_LONG: tipo de datos (garantiza precisión para muestras grandes)
	//   * MPI_SUM: operación a realizar (suma)
	//   * 0: proceso root que recibirá el resultado
	//   * MPI_COMM_WORLD: comunicador que define el grupo de procesos
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

	// Solo el proceso 0 tiene el contador global y calcula PI correctamente
	// Los demás procesos mantienen resultado.pi = 0.0
	resultado.pi = (rank == 0) ? 4.0 * total_count / samples : 0.0;

	// Impresión de resultados solo desde el proceso 0
	if (rank == 0) {
		printf("----------------MPI MonterCarlo Paralelizado----------------\n");
		printf("Numero de Samples = %lld\n", samples);
		printf("Numero de Procesos = %d\n", size);
		printf("pi = %.12f\n", resultado.pi);
		printf("Tiempo de ejec./elemento de calculo (en segundos) => %.12lf s\n", resultado.tiempo_segundos);
		printf("Tiempo de ejec./elemento de calculo (en milisegundos) => %.8lf ms\n", resultado.tiempo_ms);
		printf("Tiempo de ejec./elemento de calculo (en microsegundos) => %.8lf us\n", resultado.tiempo_us);

		printf("Error absoluto: %.12f (%.8f%%)\n",
			fabs(resultado.pi - M_PI),
			100.0 * fabs(resultado.pi - M_PI) / M_PI);
	}

	return resultado;
}

// Exporta los resultados de ambas versiones a un archivo CSV (sobrescribe en cada ejecución)
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

		// Secuencial
		csv << "Secuencial;" << secuencial.samples << ";";
		csv.imbue(std::locale("es_ES"));
		csv << std::fixed << std::setprecision(12) << secuencial.pi << ";"
			<< std::fixed << std::setprecision(12) << secuencial.tiempo_segundos << ";"
			<< std::fixed << std::setprecision(8) << secuencial.tiempo_ms << ";"
			<< std::fixed << std::setprecision(8) << secuencial.tiempo_us << ";";
		csv.imbue(std::locale::classic());
		csv << secuencial.num_procesos << "\n";

		// MPI
		csv << "MPI;" << mpi.samples << ";";
		csv.imbue(std::locale("es_ES"));
		csv << std::fixed << std::setprecision(12) << mpi.pi << ";"
			<< std::fixed << std::setprecision(12) << mpi.tiempo_segundos << ";"
			<< std::fixed << std::setprecision(12) << mpi.tiempo_ms << ";"
			<< std::fixed << std::setprecision(12) << mpi.tiempo_us << ";";
		csv.imbue(std::locale::classic());
		csv << mpi.num_procesos << "\n";

		csv.close();
	}
}


// Programa principal: ejecuta las pruebas para varios tamaños de muestra
int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Array con los diferentes tamaños de muestras a probar
	const long long sample_sizes[] = {
		1000,           // 1 mil - evaluación muy rápida
		5000,           // 5 mil
		10000,          // 10 mil - evaluación rápida
		50000,          // 50 mil
		100000,         // 100 mil
		500000,         // 500 mil
		1000000,        // 1 millón - buena precisión
		5000000,        // 5 millones
		10000000,       // 10 millones - alta precisión
		50000000        // 50 millones - muy alta precisión
	};
	const int num_iterations = 10;

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
