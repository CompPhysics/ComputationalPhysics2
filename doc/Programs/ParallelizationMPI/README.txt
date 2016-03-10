For linux/ubuntu users, you need to install two packages (alternatively use the synaptic package manager)
o  sudo apt-get install libopenmpi-dev
o  sudo apt-get install openmpi-bin

For OS X users, install brew (after having installed xcode and gcc, needed for the 
gfortran compiler of openmpi) and then run
o  brew install openmpi

When running an executable (code.x), run as
o mpirun -n 10 ./code.x

where -n indicates the number of processes, 10 here.

With openmpi installed, when using Qt, add to your .pro file the instructions at
http://dragly.org/2012/03/14/developing-mpi-applications-in-qt-creator/

You may need to tell Qt where opempi is stored.

For the machines at the computer lab, openmpi is located  at /usr/lib64/openmpi/bin
Add to your .bashrc file the following
export PATH=/usr/lib64/openmpi/bin:$PATH 


For running on SMAUG, go to http://comp-phys.net/ and click on the link internals and click on
computing cluster.
To get access to Smaug, you will need to send us an e-mail with your name, UiO username, phone number, room number and affiliation to the research group. In return, you will receive a password you may use to access the cluster.

Here follows a simple recipe
log in as
ssh -username tid.uio.no
then do an ssh to ssh username@fyslab-compphys

In the folder shared/guides/starting_jobs you will find a simple example on how to set up a job and compile and run.
This files are write protected. Copy them to your own folder and compile and run there. 
The following simple hello world c++ code
//    First C++ example of MPI Hello world
using namespace std;
#include <mpi.h>
#include <iostream>
  
int main (int nargs, char* args[])
{
     int numprocs, my_rank;
//   MPI initializations
     MPI_Init (&nargs, &args);
     MPI_Comm_size (MPI_COMM_WORLD, &numprocs);
     MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
     cout << "Hello world, I have  rank " << my_rank << " out of " << numprocs << endl;
//  End MPI
      MPI_Finalize ();
    return 0;
 }

can be compiled  as 
mpic++ -O3 -o hw1.x hw1.cpp
and you can run it as 
mpirun -n 20 ./hw1.x


For further instructions follow the help guidelines at http://comp-phys.net/cluster-info/using-smaug/
