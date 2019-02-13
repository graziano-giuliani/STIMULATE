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
  const std::size_t tnx = 4096;
  const std::size_t tny = 4096;
  const std::size_t ntimes = 5;

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

  exit(0);

  // Start out the communication
  // The IO processor sends the data to all the other processors
  
  if ( rank == iorank ) {
    std::cout << "Initial communication." << std::endl;
  }
  std::vector<int> sendcnt(nproc);
  std::vector<int> displs(nproc); 
  for (  int ir = 0; ir < nproc; ir ++ ) {
    // sendcnt[ir] = ????;
    // displs[ir] = ????;
  }

  // int MPI_Scatterv(const void *sendbuf,
  //                  const int sendcounts[],
  //                  const int displs[],
  //                  MPI_Datatype sendtype,
  //                  void *recvbuf,
  //                  int recvcount,
  //                  MPI_Datatype recvtype,
  //                  int root,
  //                  MPI_Comm comm)

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
    // hid_t H5Fcreate( const char *name,
    //                  unsigned flags,
    //                  hid_t fcpl_id,
    //                  hid_t fapl_id )
    // hsize_t dims[3] = ????????????;
    // hsize_t maxdims[3] = ????????????;
    // hsize_t chunk_dims[3] = ????????????;
    // Set chunking properties
    // hid_t H5Pcreate( hid_t cls_id )
    // herr_t H5Pset_chunk( hid_t plist,
    //                      int ndims,
    //                      const hsize_t * dim )
    // Create dataspace
    // hid_t H5Screate_simple( int rank,
    //                         const hsize_t * current_dims,
    //                         const hsize_t * maximum_dims )
    // Create dataset
    // hid_t H5Dcreate2( hid_t loc_id,
    //                   const char *name,
    //                   hid_t dtype_id,
    //                   hid_t space_id,
    //                   hid_t lcpl_id,
    //                   hid_t dcpl_id,
    //                   hid_t dapl_id )
    // H5Pclose( hid_t prop_id )
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

    // Collect the new solution on IO processor
    // int MPI_Gatherv(const void *sendbuf,
    //                       int sendcount,
    //                       MPI_Datatype sendtype,
    //                       void *recvbuf,
    //                       const int recvcounts[],
    //                       const int displs[],
    //                       MPI_Datatype recvtype,
    //                       int root,
    //                       MPI_Comm comm)

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
      // hid_t H5Dget_space( hid_t dataset_id )
      // hsize_t offset[3] = ?????????????;
      // hsize_t count[3] = ????????????;
      // Select destination hyperslab
      // herr_t H5Sselect_hyperslab( hid_t space_id,
      //                             H5S_seloper_t op,
      //                             const hsize_t *start,
      //                             const hsize_t *stride,
      //                             const hsize_t *count,
      //                             const hsize_t *block )
      // Create dataspace
      // hid_t H5Screate_simple( int rank,
      //                         const hsize_t * current_dims,
      //                         const hsize_t * maximum_dims )
      // Write data
      // herr_t H5Dwrite( hid_t dataset_id,
      //                  hid_t mem_type_id,
      //                  hid_t mem_space_id,
      //                  hid_t file_space_id,
      //                  hid_t xfer_plist_id,
      //                  const void * buf )
      std::cout << "Timestep " << timestep << std::endl;
      if ( timestep < ntimes - 1 ) {
        hsize_t newdims[3];
        // Extend dataset
        // herr_t H5Dset_extent( hid_t dataset_id,
        //                       const hsize_t size[] )
      }
      // H5Sclose( hid_t space_id )
      // H5Sclose( hid_t space_id )
    }

  } // time loop

  // The Armadillo library has a dump helper in HDF5 format ;)
  //if ( rank == iorank ) {
  //  tpm.save(arma::hdf5_name("tpm.h5", "tpm"));
  //}

  // Resource cleanup
  // Note that processor 0 is unbalanced and has a lot more resources.
  if ( rank == iorank ) {
    // H5Dclose( hid_t dset_id );
    // H5Fclose( hid_t file_id );
    tpm.reset( );
  }
  lpm.reset( );

  (void) MPI_Finalize();
}
