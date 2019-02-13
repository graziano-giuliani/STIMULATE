
ARMA=${HOME}/stimulate

MPICXX = mpiicpc
CXXFLAGS = -std=c++17
CPPFLAGS = -I$(ARMA)/include -I${HDF5_INC}
LDFLAGS = -L${HDF5_HOME}/lib -L$(ARMA)/lib64 -Wl,-rpath=$(ARMA)/lib64
LIBS = -lhdf5 -larmadillo

model : model.cpp
	$(MPICXX) $(CXXFLAGS) -o $@ $(CPPFLAGS) $(LDFLAGS) $< $(LIBS)

clean :
	rm -f model
