
ARMA=/usr
HDF5_HOME=/usr
HDF5_INC=/usr/include/hdf5/openmpi

MPICXX = mpic++
CXXFLAGS = -std=c++17
CPPFLAGS = -I$(ARMA)/include -I${HDF5_INC}
LDFLAGS = -L${HDF5_HOME}/lib -Wl,-rpath=$(ARMA)/lib
LIBS = -lhdf5_openmpi -larmadillo

benchmark_LBE3D_IO.x : benchmark_LBE3D_IO.c
	$(MPICXX) $(CXXFLAGS) -o $@ $(CPPFLAGS) $(LDFLAGS) $< $(LIBS)

clean :
	rm -f benchmark_LBE3D_IO.x
