//
// STIMULATE - Gather data and write
//
// Excercise is create the output HDF5 file and write all the
// "model" timestep to the output file.
// You have as "pseudocode" the HDF5 functions you should call.
//

#include <sys/time.h>
#include <armadillo>
#include <iostream>
#include <numeric>
#include <vector>
#include <mpi.h>
#include <hdf5.h>

/* Return the second elapsed since Epoch (00:00:00 UTC, January 1, 1970)
 */
double time_in_seconds()
{
  struct timeval tmp;
  double sec;
  gettimeofday( &tmp, (struct timezone *)0 );
  sec = tmp.tv_sec + ((double)tmp.tv_usec)/1000000.0;
  return sec;
}

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

  double tot_time_io = 0.0, t_start;

  // Global problem size (program arguments)
  const hsize_t tnx = 512;
  const hsize_t tny = 256;
  const hsize_t ntimes = 10;

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
     hsize_t xmiss = tnx - nx * nproc;
     if ( rank < xmiss ) {
       xstart = xstart + rank;
       nx = nx + 1;
     } else {
       xstart = xstart + xmiss;
     }
  }

  // Let all processors have clear view of model topology
  std::vector<hsize_t> gsx(nproc);
  std::vector<hsize_t> gnx(nproc);
  {
    (void) MPI_Allgather(&xstart,1,MPI_LONG,gsx.data( ),1,MPI_LONG,global_comm);
    (void) MPI_Allgather(&nx,1,MPI_LONG,gnx.data( ),1,MPI_LONG,global_comm);
  }

  // Allocate the initial local matrix
  auto lpm = arma::Mat<md_type>(ny,nx);

  
  auto lpm_io = arma::Mat<md_type>();
  if ( rank == iorank ) {
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
    hsize_t dims[3] = {1L,tnx,tny};
    hsize_t maxdims[3] = {H5S_UNLIMITED,tnx,tny};
    hsize_t chunk_dims[3] = {1L,tnx,tny};
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
    for ( hsize_t iloop = 0; iloop < 100; iloop ++ ) {
      for ( hsize_t i = 0; i < nx; i ++ ) {
        for ( hsize_t j = 0; j < ny; j ++ ) {
          lpm(j,i) = (md_type) rank + timestep;
        }
      }
    }

    t_start = time_in_seconds();
    if ( rank == iorank ) { 
      
      // Write new solution as timestep in the HDF5 file
      for ( hsize_t ir = 0; ir < nproc; ir ++ ) {
	
	md_type * tmp_ptr;
	
	if ( ir == iorank ) {
	  tmp_ptr = lpm.memptr();
	}
	else {
	  tmp_ptr = lpm_io.memptr();
	  MPI_Recv( tmp_ptr, gnx[ ir ] * tny,
                    md_mpi_type, ir, 100, global_comm, MPI_STATUS_IGNORE );
	}
	
	// Get filespade
	filespace = H5Dget_space(dataset);
	hsize_t offset[3] = { timestep,gsx[ir],0L };
	hsize_t count[3] = { 1L,gnx[ ir ],tny };
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
      }

      std::cout << "Timestep " << timestep << std::endl;
      if ( timestep < ntimes - 1 ) {
	hsize_t newdims[3] = { timestep+2,tnx,tny };
	// Extend dataset
	status = H5Dset_extent(dataset, newdims);
      }

    }
    else { 
      MPI_Send( lpm.memptr(), nx * ny, md_mpi_type, 0, 100, global_comm );
    }
    
    tot_time_io += time_in_seconds() - t_start;
    
  } // time loop

  if ( rank == iorank ) {
    double m_bytes = ( ( double) tnx*tny*sizeof( md_type )*ntimes )/1e6;
    std::cout << std::endl << std::endl
              << "\t\tNumber of MPI processes = " << nproc << std::endl
              << "\t\tTime of I/O = " << tot_time_io << " seconds" << std::endl
              << "\t\tSize of the data = " << m_bytes << " MBytes" << std::endl
              << "\t\tI/O bandwidth = " << m_bytes / tot_time_io
              << " MBytes/sec" << std::endl << std::endl;
  }

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
