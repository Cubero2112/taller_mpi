## Ejecucion de un programa por medio de MPI (Message Passing Interface)

El desarrollo de este programa se realizo en base a [MPI Tutorial](https://mpitutorial.com/). En este sitio web se encuentra detalladamente la configuracion detallada del cluster en una LAN. 
### Requerimientos del sistema

- Compilador g++
- Compilador gcc
- Herramienta de build automatico: make
- MPICH


### Ejecucion del codigo

* La ejecucion debe de realizarse desde el directorio cloud/array_ops
* Se compila con el comando: mpicc -o array_sum array_sum.c
* La ejecucion se realiza con el comando: mpirun -np 2 -hosts <IP_00>,<IP_01> ./array_sum
	* np: Número de procesos en los cuales se de
	* host: Incluye las máquinas sobre las cuales se distribuye la carga de trabajo, en este caso se utiliza la máquina Manager y Worker.

