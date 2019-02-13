//
// STIMULATE - Gather data and write
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
// Utility function
//
// Aim is to find the two largest numbers that give as product np
//
void split_proc(int np, std::vector<int> &bd)
{
  std::vector<int> factors;
  int num = np;
  int maxf = 2;
  while ( num >= maxf ) {
    while ( num % maxf == 0 ) {
      factors.insert(factors.begin( ),maxf);
      num = num / maxf;
    }
    maxf = maxf + 1;
  }
  num = 0;
  std::fill(bd.begin(), bd.end(), 1);
  while ( ! factors.empty( ) ) {
    auto pp = std::accumulate(std::begin(bd),
                              std::end(bd), 1, std::multiplies<int>());
    if ( pp > 1 && factors.back( ) > pp/2 ) {
      std::fill(bd.begin(), bd.end(), 1);
      bd[0] = pp;
      bd[1] = factors.back( );
    } else {
      bd[num] = bd[num] * factors.back( );
    }
    factors.pop_back( );
    num = num + 1;
    if ( num == (int) bd.size( ) ) num = 0;
  } 
}

//
// Main program
//
int main(int argc, char *argv[])
{
  // Global problem size (program arguments)
  const std::size_t tnx = 200;
  const std::size_t tny = 100;
  const std::size_t tnz = 10;
  const std::size_t ntimes = 10;

  // Parallelize over first two dimensions
  const int ndims = 2;

  // Two ghost points. Stencil dependent
  const int nghost = 2;

  // Utility
  enum { left = 0, top, right, bottom };

  // Processor ghost points and neighbours
  std::size_t gpa[4] = { 0, 0, 0, 0 };
  int gneigh[4] = { MPI_PROC_NULL, MPI_PROC_NULL,
                    MPI_PROC_NULL, MPI_PROC_NULL };

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

  // Decomposition : find two largest integers such that
  // nproc = cpus_per_dim[0] * cpus_per_dim[1]
  std::vector<int> cpus_per_dim(ndims);
  split_proc(nproc,cpus_per_dim);

#ifdef DEBUG
  if ( rank == iorank ) {
    std::cout << "Decomposition : " << cpus_per_dim[0] <<
                 " x " << cpus_per_dim[1] << std::endl;
    std::cout << "Processor " << rank << ": Global storage of "
             << tnx * tny * tnz * sizeof(md_type) << " bytes" << std::endl;
  }
#endif

  // Define problem topology with chosen 2D decomposition

  MPI_Comm cart_comm;
  int crank;
  std::vector<int> coords;
  {
    std::vector<int> periods(ndims);
    std::fill(periods.begin(), periods.end(), 0);
    (void) MPI_Cart_create(global_comm, ndims, cpus_per_dim.data( ),
                           periods.data( ), 0, &cart_comm);
    (void) MPI_Comm_rank(cart_comm, &crank);
    coords.resize(ndims);
    std::fill(coords.begin(), coords.end(), -1);
    (void) MPI_Cart_coords(cart_comm, crank, ndims, coords.data());

    std::vector<int> isearch = { coords[0]-1, coords[1] };
    if ( isearch[0] >= 0 ) {
      (void) MPI_Cart_rank(cart_comm, isearch.data(), &gneigh[left]);
    }
    isearch = { coords[0], coords[1]+1 };
    if ( isearch[1] < cpus_per_dim[1]) {
      (void) MPI_Cart_rank(cart_comm, isearch.data(), &gneigh[top]);
    }
    isearch = { coords[0]+1, coords[1] };
    if ( isearch[0] < cpus_per_dim[0]) {
      (void) MPI_Cart_rank(cart_comm, isearch.data(), &gneigh[right]);
    }
    isearch = { coords[0], coords[1]-1 };
    if ( isearch[1] >= 0 ) {
      (void) MPI_Cart_rank(cart_comm, isearch.data(), &gneigh[bottom]);
    }
    if ( gneigh[left] != MPI_PROC_NULL ) gpa[left] = nghost;
    if ( gneigh[top] != MPI_PROC_NULL ) gpa[top] = nghost;
    if ( gneigh[right] != MPI_PROC_NULL ) gpa[right] = nghost;
    if ( gneigh[bottom] != MPI_PROC_NULL ) gpa[bottom] = nghost;
  }

  // Total problem memory
  auto tpm = arma::Cube<md_type>( );
  // Space allocation only on IO-Rank
  if ( crank == iorank ) {
    tpm.set_size(tny,tnx,tnz);
  }

  // Better fit the problem: assign leftover row/columns to low rank CPUs
  // (max 1 col and 1 row each)
  std::size_t nx = tnx / cpus_per_dim[0];
  auto xstart = coords[0]*nx;
  if ( nx * cpus_per_dim[0] != tnx ) {
     int imiss = tnx - nx * cpus_per_dim[0];
     if ( coords[0] < imiss ) {
       xstart = xstart + coords[0];
       nx = nx + 1;
     } else {
       xstart = xstart + imiss;
     }
  }
  std::size_t ny = tny / cpus_per_dim[1];
  auto ystart = coords[1]*ny;
  if ( ny * cpus_per_dim[1] != tny ) {
     int ymiss = tny - ny * cpus_per_dim[1];
     if ( coords[1] < ymiss ) {
       ystart = ystart + coords[1];
       ny = ny + 1;
     } else {
       ystart = ystart + ymiss;
     }
  }
  std::size_t nz = tnz;

  // Processor local memory
  auto lpm = arma::Cube<md_type>(ny+gpa[top]+gpa[bottom],
                                 nx+gpa[left]+gpa[right],
                                 nz);
  // Guard value to check communications
  lpm.fill(-100.0);

  // Let all processors have clear view of model topology
  std::vector<int> gip(nproc);
  std::vector<int> gsx(nproc);
  std::vector<int> gsy(nproc);
  std::vector<int> gsz(nproc);
  std::vector<int> gnx(nproc);
  std::vector<int> gny(nproc);
  std::vector<int> gnz(nproc);

  for ( int ir = 0; ir < nproc; ir ++ ) {
    gsz[ir] = 0;
    gnz[ir] = tnz;
  }

  {
    (void) MPI_Allgather(&crank,1,MPI_INT,gip.data( ),1,MPI_INT,cart_comm);
    (void) MPI_Allgather(&xstart,1,MPI_INT,gsx.data( ),1,MPI_INT,cart_comm);
    (void) MPI_Allgather(&ystart,1,MPI_INT,gsy.data( ),1,MPI_INT,cart_comm);
    (void) MPI_Allgather(&nx,1,MPI_INT,gnx.data( ),1,MPI_INT,cart_comm);
    (void) MPI_Allgather(&ny,1,MPI_INT,gny.data( ),1,MPI_INT,cart_comm);
  }

  std::vector<int> gex(nproc);
  std::vector<int> gey(nproc);
  std::vector<int> gez(nproc);
  for ( int ir = 0; ir < nproc; ir ++ ) {
    gex[ir] = gsx[ir] + gnx[ir] - 1;
    gey[ir] = gsy[ir] + gny[ir] - 1;
    gez[ir] = gsz[ir] + gnz[ir] - 1;
  }

#ifdef DEBUG
  if ( crank == iorank ) {
    for ( int ir = 0; ir < nproc; ir ++ ) {
      std::cout << "Processor " << ir << ": Local storage : " 
           << gnx[ir]*gny[ir]*gnz[ir]*sizeof(md_type) << " bytes "
           << "starting at : (" 
           << gsx[ir] << "," << gsy[ir] << "," << gsz[ir] << ") " 
           << "ending at : ( " 
           << gex[ir] << "," << gey[ir] << "," << gez[ir] << ")"
           << std::endl;
     }
   }
#endif

  // Fill total problem memory using armadillo column ordering of cubes.
  // We want each processor to receive constant values equal to
  // its rank + 1
  if ( crank == iorank ) {
    for ( int ir = 0; ir < nproc; ir ++ ) {
      for ( arma::uword k = gsz[ir]; k <= gez[ir]; k ++ ) {
        for ( arma::uword i = gsx[ir]; i <= gex[ir]; i ++ ) {
          for ( arma::uword j = gsy[ir]; j <= gey[ir]; j ++ ) {
            tpm(j,i,k) = (md_type) (gip[ir]+1);
          }
        }
      }
    }
  }

  // Set up global communicator type.
  std::vector<MPI_Datatype> glob_type(nproc);
  std::vector<MPI_Request> reqs;
 
  // The IO Processor has to know where each local procesor memory block
  // is inside the global problem memory.
  if ( crank == iorank ) {
    glob_type.resize(nproc);
    int aosize[3] = { tnz, tnx, tny };
    for ( int ir = 0; ir < nproc; ir ++ ) {
      int aosubsize[3] = { gnz[ir], gnx[ir], gny[ir] };
      int aostart[3] = { gsz[ir], gsx[ir], gsy[ir] };
      (void) MPI_Type_create_subarray(3, aosize, aosubsize, aostart,
                             MPI_ORDER_C, md_mpi_type, &glob_type[ir]);
      (void) MPI_Type_commit(&glob_type[ir]);
    }
  }

  // Local processor memory has to take into account the ghost point
  // for the computational stencil. This part is just to have a realistic
  // communication pattern set in place similar to what is usual for an
  // atmospheric model dynamical solver.
  MPI_Datatype loc_type;
  int losize[3] = { (int) nz,
                    (int) (nx+gpa[left]+gpa[right]),
                    (int) (ny+gpa[top]+gpa[bottom])};
  int losubsize[3] = { (int) nz, (int) nx, (int) ny };
  int lostart[3] = { 0, (int) gpa[left], (int) gpa[bottom] };
  (void) MPI_Type_create_subarray(3, losize, losubsize, lostart,
                           MPI_ORDER_C, md_mpi_type, &loc_type);
  (void) MPI_Type_commit(&loc_type);

  // Start out the communication

  // The IO processor sends the data to all the other processors
  if ( crank == iorank ) {
    reqs.resize(nproc);
    for ( int ir = 0; ir < nproc; ir ++ ) {
      (void) MPI_Isend(tpm.memptr(), 1, glob_type[ir],
                       ir, ir+100, cart_comm, &reqs[ir]);
    }
  }
  // All of them receive it,
  (void) MPI_Recv(lpm.memptr( ), 1, loc_type, iorank, crank+100,
                  cart_comm, MPI_STATUS_IGNORE);
  // Communication completed.
  if ( crank == iorank ) {
    (void) MPI_Waitall(nproc, reqs.data( ), MPI_STATUSES_IGNORE);
  }

  // Check if communication has been performed : all results should be zero
  if ( int(lpm(gpa[bottom],gpa[left],0)) - crank != 1 ) {
    std::cerr << int(lpm(gpa[bottom],gpa[left],0))
           << " : SEND on proc " << crank << std::endl;
    (void) MPI_Abort(cart_comm,-1);
  }

  // Reset global storage space
  if ( crank == iorank ) {
    tpm.fill(0.0);
  }

  // IO of data using HDF5 library

  hid_t file, dataset, dataspace;
  hid_t filespace, prop;
  herr_t status;

  // Only processor 0 has to open the file with this strategy

  if ( crank == iorank ) {
    file = H5Fcreate(FILE_NAME, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hsize_t dims[4] = {1,tnz,tnx,tny};
    hsize_t maxdims[4] = {H5S_UNLIMITED,tnz,tnx,tny};
    hsize_t chunk_dims[4] = {1,tnz,tnx,tny};
    prop = H5Pcreate(H5P_DATASET_CREATE);
    status = H5Pset_chunk(prop, 4, chunk_dims);
    dataspace = H5Screate_simple(4, dims, maxdims);
    dataset = H5Dcreate2(file, "/tpm", md_h5_type, dataspace,
                         H5P_DEFAULT, prop, H5P_DEFAULT);
    status = H5Pclose(prop);
  }

  // simulate a model running and compute the new solution

  for ( hsize_t timestep = 0; timestep < ntimes; timestep ++ ) {

    // Fake use of CPU time
    for ( int iloop = 0; iloop < 1000; iloop ++ ) {
      for ( int k = 0; k < nz; k ++ ) {
        lpm(arma::span(gpa[bottom],ny+gpa[bottom]-1),
            arma::span(gpa[left],nx+gpa[left]-1),
            arma::span(k,k)).fill((md_type) k+crank+timestep);
       }
    }

    // Collect the new solution on IO processor
    if ( crank == iorank ) {
      for ( int ir = 0; ir < nproc; ir ++ ) {
        (void) MPI_Irecv(tpm.memptr(), 1, glob_type[ir],
                         ir, ir+100, cart_comm, &reqs[ir]);
      }
    }
    (void) MPI_Send(lpm.memptr( ), 1, loc_type,
                    iorank, crank+100, cart_comm);
    if ( crank == iorank ) {
      (void) MPI_Waitall(nproc, reqs.data( ), MPI_STATUSES_IGNORE);
    }

    // Assert communication is ok
    if ( crank == iorank ) {
      for ( int ir = 0; ir < nproc; ir ++ ) {
        if ( int(tpm(gsy[ir],gsx[ir],gsz[ir])) != gip[ir] + timestep ) {
          std::cerr << int(tpm(gsy[ir],gsx[ir],gsz[ir])) << " " << gip[ir] + 2
                    << " : ERROR on proc " << crank << std::endl;
          (void) MPI_Abort(cart_comm,-1);
        }
      }
    }

    // Write new solution as timestep in the HDF5 file
    if ( crank == iorank ) {
      filespace = H5Dget_space(dataset);
      hsize_t offset[4] = { timestep,0,0,0 };
      hsize_t count[4] = { 1,tnz,tnx,tny };
      status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL,
                                   count, NULL);
      dataspace = H5Screate_simple(4, count, NULL);
      status = H5Dwrite(dataset, md_h5_type, dataspace, filespace,
                        H5P_DEFAULT, tpm.memptr( ));
      status = H5Sclose(filespace);
      status = H5Sclose(dataspace);
      std::cout << "Timestep " << timestep << std::endl;
      if ( timestep < ntimes - 1 ) {
        hsize_t newdims[4] = { timestep+2,tnz,tnx,tny };
        status = H5Dset_extent(dataset, newdims);
      }
    }
  } // time loop

  // Resource cleanup
  // Note that processor 0 is unbalanced and has a lot more resources.
  if ( crank == iorank ) {
    status = H5Dclose(dataset);
    status = H5Fclose(file);
    for ( int ir = 0; ir < nproc; ir ++ ) {
      (void) MPI_Type_free(&glob_type[ir]);
    }
    tpm.reset( );
  }
  (void) MPI_Type_free(&loc_type);
  (void) MPI_Comm_free(&cart_comm);
  lpm.reset( );

  (void) MPI_Finalize();
}
