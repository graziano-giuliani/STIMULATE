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

// Constants

#define md_type float
#define md_mpi_type MPI_FLOAT
#define md_h5_type H5T_NATIVE_FLOAT
#define FILE_NAME "model_out.h5"

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

//
// Main program
//
int main(int argc, char *argv[])
{
  double tot_time_io = 0.0, t_start;

  // Global problem size (program arguments)
  const std::size_t tnx = 512;
  const std::size_t tny = 256;
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

  // Total problem memory
  auto tpm = arma::Mat<md_type>( );
  // Space allocation only on IO-Rank
  if ( rank == iorank ) {
    tpm.set_size(tny,tnx);
  }

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

  // Fill total problem memory using armadillo column ordering of Matrix.
  // We want each processor to receive constant values equal to
  // its rank + 1
  if ( rank == iorank ) {
    for ( int ir = 0; ir < nproc; ir ++ ) {
      auto ex = gsx[ir] + gnx[ir];
      for ( arma::uword i = gsx[ir]; i < ex; i ++ ) {
        for ( arma::uword j = 0; j < ny; j ++ ) {
          tpm(j,i) = (md_type) (ir+1);
        }
      }
    }
  }

  // Processor local memory
  auto lpm = arma::Mat<md_type>(ny,nx);

  // Guard value to check communications
  lpm.fill(-100.0);

  // Start out the communication
  // The IO processor sends the data to all the other processors
  
  if ( rank == iorank ) {
    std::cout << "Initial communication." << std::endl;
  }
  std::vector<int> sendcnt(nproc);
  std::vector<int> displs(nproc); 
  for (  int ir = 0; ir < nproc; ir ++ ) {
    sendcnt[ir] = gnx[ir]*ny;
    displs[ir] = gsx[ir]*ny;
  }
  (void) MPI_Scatterv(tpm.memptr( ), sendcnt.data( ), displs.data( ),
                      md_mpi_type, lpm.memptr( ), ny*nx, md_mpi_type,
                      iorank, global_comm);

  // Check if communication has been performed : all results should be zero
  if ( int(lpm(0,0)) - rank != 1 ) {
    std::cerr << int(lpm(0,0))
           << " : SEND on proc " << rank << std::endl;
    (void) MPI_Abort(global_comm,-1);
  }

  if ( rank == iorank ) {
    std::cout << "Comm ok." << std::endl;
  }
  // Reset global storage space
  if ( rank == iorank ) {
    tpm.fill(0.0);
  }

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
      for ( hsize_t i = 0; i < nx; i ++ ) {
        for ( hsize_t j = 0; j < ny; j ++ ) {
          lpm(j,i) = (md_type) rank+timestep;
        }
      }
    }

    t_start = time_in_seconds();

    // Collect the new solution on IO processor
    (void) MPI_Gatherv(lpm.memptr( ), ny*nx, md_mpi_type,
                       tpm.memptr( ), sendcnt.data( ), displs.data( ),
                       md_mpi_type, iorank, global_comm);

    // Assert communication is ok
    if ( rank == iorank ) {
      for ( int ir = 0; ir < nproc; ir ++ ) {
        if ( int(tpm(0,gsx[ir])) != ir + timestep ) {
          std::cerr << int(tpm(0,gsx[ir])) << " " << ir + 2
                    << " : ERROR from proc " << ir << std::endl;
          (void) MPI_Abort(global_comm,-1);
        }
      }
    }

    // Write new solution as timestep in the HDF5 file
    if ( rank == iorank ) {
      // Get filespade
      filespace = H5Dget_space(dataset);
      hsize_t offset[3] = { timestep,0,0 };
      hsize_t count[3] = { 1,tnx,tny };
      // Select destination hyperslab
      status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL,
                                   count, NULL);
      // Create dataspace
      dataspace = H5Screate_simple(3, count, NULL);
      // Write data
      status = H5Dwrite(dataset, md_h5_type, dataspace, filespace,
                        H5P_DEFAULT, tpm.memptr( ));
      status = H5Sclose(filespace);
      status = H5Sclose(dataspace);
      std::cout << "Timestep " << timestep << std::endl;
      if ( timestep < ntimes - 1 ) {
        hsize_t newdims[3] = { timestep+2,tnx,tny };
        // Extend dataset
        status = H5Dset_extent(dataset, newdims);
      }
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

  // The Armadillo library has a dump helper in HDF5 format ;)
  //if ( rank == iorank ) {
  //  tpm.save(arma::hdf5_name("tpm.h5", "tpm"));
  //}

  // Resource cleanup
  // Note that processor 0 is unbalanced and has a lot more resources.
  if ( rank == iorank ) {
    status = H5Dclose(dataset);
    status = H5Fclose(file);
    tpm.reset( );
  }
  lpm.reset( );

  (void) MPI_Finalize();
}
