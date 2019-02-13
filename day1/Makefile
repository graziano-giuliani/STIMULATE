
ARMA=/usr
HDF5_HOME=/usr
HDF5_INC=/usr/include/hdf5/openmpi

MPICXX = mpic++
CXXFLAGS = -std=c++17
CPPFLAGS = -I$(ARMA)/include -I${HDF5_INC}
LDFLAGS = -L${HDF5_HOME}/lib -Wl,-rpath=$(ARMA)/lib
LIBS = -lhdf5_openmpi -larmadillo

model : model.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $(CPPFLAGS) $(LDFLAGS) $< $(LIBS)

clean :
	rm -f model model_out.h5
