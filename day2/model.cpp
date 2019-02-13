//
// STIMULATE - Gather data and write
//
// Excercise is create the output HDF5 file and write all the
// "model" timestep to the output file.
// You have as "pseudocode" the HDF5 functions you should call.
//

#include <armadillo>
#include <iostream>
#include <numeric>
#include <vector>
#include <mpi.h>
#include <hdf5.h>

// Constants

#define md_type float
#define md_mpi_type MPI_FLOAT
#define md_h5_type H5T_NATIVE_FLOAT
#define FILE_NAME "model_out.h5"

//
// Main program
//
int main(int argc, char *argv[])
{
  // Global problem size (program arguments)
  const std::size_t tnx = 2000;
  const std::size_t tny = 1000;
  const std::size_t ntimes = 10;

  // Exercise target : gather things here
  const int iorank = 0;

  // MPI part
  MPI_Comm global_comm;
  int nproc , rank;

  // Standard MPI initialization
  (void) MPI_Init(&argc,&argv);
  (void) MPI_Comm_dup(MPI_COMM_WORLD, &global_comm);
  (void) MPI_Comm_size(global_comm, &nproc);
  (void) MPI_Comm_rank(global_comm, &rank);

  // Local sizes
  auto nx = tnx / nproc;
  auto ny = tny;

  // Better fit the problem: assign leftover rows to the CPUs
  // (max 1 row each)
  auto xstart = rank*nx;
  if ( nx * nproc != tnx ) {
     int xmiss = tnx - nx * nproc;
     if ( rank < xmiss ) {
       xstart = xstart + rank;
       nx = nx + 1;
     } else {
       xstart = xstart + xmiss;
     }
  }

  // Let all processors have clear view of model topology
  std::vector<int> gsx(nproc);
  std::vector<int> gnx(nproc);
  {
    (void) MPI_Allgather(&xstart,1,MPI_INT,gsx.data( ),1,MPI_INT,global_comm);
    (void) MPI_Allgather(&nx,1,MPI_INT,gnx.data( ),1,MPI_INT,global_comm);
  }

  // Allocate the initial local matrix
  auto lpm = arma::Mat<md_type>(ny,nx);

  
  auto lpm_io = arma::Mat<md_type>();
  if( rank == iorank ){
    lpm_io.resize( ny, nx );
  }

  // Initialization of the matrix 
  lpm.fill( (md_type) rank );

  // IO of data using HDF5 library
  hid_t file, dataset, dataspace;
  hid_t filespace, prop;
  herr_t status;

  // Only processor 0 has to open the file with this strategy

  if ( rank == iorank ) {
    // Create the file
    file = H5Fcreate(FILE_NAME, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hsize_t dims[3] = {1,tnx,tny};
    hsize_t maxdims[3] = {H5S_UNLIMITED,tnx,tny};
    hsize_t chunk_dims[3] = {1,tnx,tny};
    // Set chunking properties
    prop = H5Pcreate(H5P_DATASET_CREATE);
    status = H5Pset_chunk(prop, 3, chunk_dims);
    // Create dataspace
    dataspace = H5Screate_simple(3, dims, maxdims);
    // Create dataset
    dataset = H5Dcreate2(file, "/tpm", md_h5_type, dataspace,
                         H5P_DEFAULT, prop, H5P_DEFAULT);
    H5Pclose(prop);
  }

  // simulate a model running and compute the new solution
  for ( hsize_t timestep = 0; timestep < ntimes; timestep ++ ) {

    // Fake use of CPU time
    for ( int iloop = 0; iloop < 100; iloop ++ ) {
      for ( int i = 0; i < nx; i ++ ) {
        for ( int j = 0; j < ny; j ++ ) {
          lpm(j,i) = (md_type) rank + timestep;
        }
      }
    }

    if( rank == iorank ){ 
      
      // Write new solution as timestep in the HDF5 file
      for ( int ir = 0; ir < nproc; ir ++ ) {
	
	// pointer to the data to write
	md_type * tmp_ptr;
	
	if( ir == iorank ){
          // write own memory
        } else {
	  // Implement the receive side from other procs
        }
        
	// Get filespace
	filespace = H5Dget_space(dataset);
	  
	// Define the HDF5 the hyperslab offset and count accordingly 
	//	  hsize_t offset[3] = ???
	//	  hsize_t count[3] = ???
	// Select destination hyperslab
	status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL,
				     count, NULL);
	// Create dataspace
	dataspace = H5Screate_simple(3, count, NULL);
	// Write data
	status = H5Dwrite(dataset, md_h5_type, dataspace, filespace,
			  H5P_DEFAULT, tmp_ptr);
	status = H5Sclose(filespace);
	status = H5Sclose(dataspace);   

	std::cout << "Timestep " << timestep << std::endl;
	if ( timestep < ntimes - 1 ) {
	  hsize_t newdims[3] = { timestep+2,tnx,tny };
	  // Extend dataset
	  status = H5Dset_extent(dataset, newdims);
	}

    }
    else { 
      // Implement the sender code
    }
    
  } // time loop

  // The Armadillo library has a dump helper in HDF5 format ;)
  //if ( rank == iorank ) {
  //  tpm.save(arma::hdf5_name("tpm.h5", "tpm"));
  //}

  // Resource cleanup
  // Note that processor 0 is unbalanced and has a lot more resources.
  if ( rank == iorank ) {
    status = H5Dclose(dataset);
    status = H5Fclose(file);
    lpm_io.reset( );
  }
  lpm.reset( );

  (void) MPI_Finalize();
}
