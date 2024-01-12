all:
	g++ -O3 -g -o scft -fPIC -mcmodel=medium modelc_scft.cpp -lm -lfftw3

debug:
	g++ -w -g -o scft modelc_scft.cpp -lm -lfftw3 -L/usr/lib
clean:
	rm -f scft *.dat *.o
