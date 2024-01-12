////
//
// SCFT.cpp
// Numerical solution to polymeric field theory in the mean field.
// Solution to Model C in "The Equilibrium Theory of Inhomogeneous Polymers"
// by Glenn H. Fredrickson
//
//
// Requires FFTW3 for Fourier transforms!
// 
// Michael J. A. Hore
// University of Pennsylvania
// June 2011
////

#include <complex>
#include <fstream>
#include <iostream>
#include "fftw3.h"
#include "stdio.h"
#include "stdlib.h"

#define PI 3.14159265358979323846263383
#define S_MESH_A 80
#define S_MESH_B 80
#define R_MESH 64
#define R_PTS R_MESH*R_MESH
#define VER_MAJ 1
#define VER_MIN 1
#define VER_REV 1
#define MAX_ITERATIONS 5000000
#define PRINT_FREQ 1

using namespace std;

static int inext,inextp;                    // Ran3
static long ma[56];                         // Ran3
static int iff=0;                           // Ran3
static int iseed=-time(0);                  // Random seed
#include "ran3.c"

int NxNyNz[3];                              // Dimension information for FFTW.
double dt;                                  // Delta t for updating fields.
double dsa, dsb;                            // Delta s in the polymer chain contour.
double lambda_p;                            // Pressure lambda for updating fields.
double lambda_x;                            // Volume lambda for updating fields.
double chiAB;                               // Flory "chi" between A and B.
double phiA_total;                          // Total (i.e., average) volume fraction of polymer A.     
double rgArgB;                              // Ratio of Rg for A to Rg for B.
double Lx;                                  // X dimension of the simulation space.
double Ly;                                  // Y dimension of the simulation space.
double Lz;				    // Z dimension of the simulation space.
double V;                                   // System volume.
complex<double> I;                          // Sqrt(-1).
complex<double> QA;                         // Single chain partition function for 'A'
complex<double> QB;                         // Single chain partition function for 'B'
complex<double> QAold;                      // QA from last iteration.
complex<double> QBold;                      // QB from last iteration.
complex<double> qA[R_PTS][S_MESH_A];	    // A propagator.
complex<double> qB[R_PTS][S_MESH_B];	    // B propagator.
complex<double> wp[R_PTS];          	    // Pressure field (W+)
complex<double> wx[R_PTS];          	    // Exchange field (W-)
complex<double> pA[R_PTS];          	    // Polymer A volume fraction.
complex<double> pB[R_PTS];          	    // Polymer B volume fraction.
complex<double> H_now, H_prev;              // Current and previous Hamiltonians.
double H_error;                             // Error in calculation. dH should be near 0 at mean field.

// Function prototypes.
void            calc_a_density(void);
void            calc_b_density(void);
complex<double> calc_error(void);
complex<double> calc_hamiltonian(void);
void            debye_gA(complex<double> *g);
void            debye_gB(complex<double> *g);
void            pseudospectral_a(complex<double> *w);
void            pseudospectral_b(complex<double> *w);
complex<double> rectangle_integrate_qA(void);
complex<double> rectangle_integrate_qB(void);
complex<double> simpson_integrate  (int arr_size, complex<double>*array, double h);
void            update_fields(void);
void            update_fields_explicit(void);
void            write_out(int);


// Main entry point
int main (void) {
	int i,j,k, r;
	FILE *error_out;

	// Parameters for calculation:
	Lx         = 10.0;
	Ly         = 10.0;
	Lz         = 10.0;
	chiAB      = 10.0;
	lambda_p   = 1.00;
	lambda_x   = 1.00;
        dt         = 1.00;
        rgArgB     = S_MESH_A / S_MESH_B;
	phiA_total = 0.30;


	// Code information.
	cout << "\nNumerical solution of polymeric field theory (Model C w/ SCFT)\n";
	cout << "Version " << VER_MAJ << "." << VER_MIN << "." << VER_REV << "\n";
	cout << "Michael J. A. Hore, University of Pennsylvania\n";
	cout << "09-Jun-2011 / 21-Jun-2011\n\n"; 

	// Initialize values that we will need later.
	real(I) = 0.0;
	imag(I) = 1.0;
	dsa     = 1.0 / double(S_MESH_A - 1);
	dsb     = 1.0 / double(S_MESH_B - 1);
	V       = Lx * Ly * Lz; 
	NxNyNz[0] = R_MESH;
	NxNyNz[1] = R_MESH;

	// Initialize fields to random values.
	for (j=0; j<R_MESH; j++) {
		for (k=0; k<R_MESH; k++) {
			r = j*R_MESH+k;
			wx[r] =  0.1*ran3(&iseed); 
			wp[r] = I*0.01; 
		}
	}

	// Write parameters to stdout.
	cout << "Using the following parameters:\n";
	cout << "Box size: " << Lx << " x " << Ly << " x " << Lz << endl;
	cout << "Rg_A / Rg_B: " << rgArgB << endl << endl;

	// Get initial fields, densities, etc.
	calc_a_density();
	calc_b_density();

	// Get initial value of Hamiltonian.
	H_prev = calc_hamiltonian();
	
	for (i=0; i<MAX_ITERATIONS; i++) {

		// Update the W+ and W- fields.
		update_fields();

		// Get initial fields, densities, etc.
		calc_a_density();
		calc_b_density();

		H_now = calc_hamiltonian();
		H_error = norm(H_now - H_prev);
		H_now = calc_error();

		if (i % PRINT_FREQ == 0) {
			cout << "QA: " << QA << "   QB: " << QB << endl;
			cout << i << " Error: " << norm(H_now - H_prev) << endl;

			// Write out to disk.
			write_out(i);

			// Keep track of errors.
			error_out = fopen("error.dat", "aw");
			fprintf(error_out, "%d %lf\n", i, H_error);
			fclose(error_out);
		}

		H_prev = H_now;

		if (H_prev != H_prev) {
			cout << "Fatal error. Exiting..." << endl;
			return 1;
		}

		if (H_error < 1.0E-13 && i > 300) {
			cout << "Solution found with superb bad-assery. Exiting successfully..." << endl;
			break;
		}

	}
	// Write out to disk.
	write_out(MAX_ITERATIONS);

	return 0;
}

//
// calc_a_density - Solve the propagator equation using a pseudospectral method, and then calculate
//                  the a-density across the system.
void calc_a_density(void) {
	int i, s;
	complex<double> *wa, *q_qdag;
	complex<double> normalize;

	// Create the A field (which is a combination of wp and wx).
	wa     = (complex<double>*) fftw_malloc(R_PTS*sizeof(complex<double>));
	q_qdag = (complex<double>*) fftw_malloc(R_PTS*sizeof(complex<double>));

	for (i=0; i<R_PTS; i++) {
		wa[i] = wp[i] - wx[i];
		//wa[i] = I*wp[i] - wx[i];
	}

	// Solve diffusion equation pseudospectrally.
	pseudospectral_a(wa);

	// Integrate qA for the polymer density.
	normalize = (phiA_total / QA);
	for (i=0; i<R_PTS; i++) {
		for (s=0; s<S_MESH_A; s++) {
			q_qdag[s] = qA[i][S_MESH_A-1-s]*qA[i][s];
		}
		pA[i] = simpson_integrate(S_MESH_A, q_qdag, dsa) * normalize;
	}

	fftw_free(wa);
	fftw_free(q_qdag);
	return;
}


//
// calc_b_density - Solve the propagator equation using a pseudospectral method, and then calculate
//                  the a-density across the system.
void calc_b_density(void) {
	int i,s;
	complex<double> *wb, *q_qdag;
	complex<double> normalize;

	// Create the A field (which is a combination of wp and wx).
	wb     = (complex<double>*) fftw_malloc(R_PTS*sizeof(complex<double>));
	q_qdag = (complex<double>*) fftw_malloc(R_PTS*sizeof(complex<double>));

	for (i=0; i<R_PTS; i++) {
		wb[i] = wp[i] + wx[i];
		//wb[i] = I*wp[i] + wx[i];
	}

	// Solve diffusion equation pseudospectrally.
	pseudospectral_b(wb);

	// Integrate qB for the polymer density.
	normalize = (1.00-phiA_total)/QB;
	for (i=0; i<R_PTS; i++) {
		for (s=0; s<S_MESH_B; s++) {
			q_qdag[s] = qB[i][S_MESH_B-1-s]*qB[i][s];
		}
		pB[i] = simpson_integrate(S_MESH_B, q_qdag, dsb) * normalize;
	}

	fftw_free(q_qdag);
	fftw_free(wb);
	return;
}

//
// calc_error - Calculates true error.
//
complex<double> calc_error(void) {
	complex<double> ret_error;
	int i;

	real(ret_error) = 0.00;
	imag(ret_error) = 0.00;

	for (i=0; i<R_PTS; i++) {
		real(ret_error) += fabs(1.00 - real(pA[i]) - real(pB[i])); 
	}

	real(ret_error) /= double(R_PTS);
	imag(ret_error) = 0.00;

	printf("Error: %lf\n", real(ret_error));
	return ret_error;
}
// 
// calc_hamiltonian - Evaluates the Hamiltonian for the system.
//
complex<double> calc_hamiltonian(void) {
	complex<double> integral, hamiltonian;
	int i;

	for (i=0; i<R_PTS; i++) {
		integral += (wx[i]*wx[i]/chiAB) - wp[i];
		//integral += (wx[i]*wx[i]/chiAB) - I*wp[i];
	}
	integral *= (V/R_PTS);

	hamiltonian = integral - phiA_total*V*log(QA) - (1.00-phiA_total)*V*log(QB);

	return hamiltonian;
}

//
// debye_gA - Calculates the Debye function for homopolymer A.
//
void debye_gA (complex<double> *g) {
	int i,j,k,x;
	double k2, kx, ky, kz;

 	for (j=0; j<R_MESH; j++) {
 		for (k=0; k<R_MESH; k++) {
			if (j > R_MESH/2.0) {
				kx = 2*PI*double(R_MESH-j)/Lx;
			} else {
				kx = 2*PI*double(j)/Lx;
			}

			if (k > R_MESH/2.0) {
				ky = 2*PI*double(R_MESH-k)/Ly;
			} else {
				ky = 2*PI*double(k)/Ly;
			}

			k2 = kx*kx + ky*ky;

			x  = k*R_MESH + j;
			if (x == 0) {
				g[x] = 1.0;
			} else {
				g[x] = 2.0*(exp(-rgArgB*k2) + k2 - 1.0) / k2 / k2;
			}
		}	
	}	
	return;
}


//
// debye_gB - Calculates the Debye function for homopolymer B.
//
void debye_gB (complex<double> *g) {
	int i,j,k,x;
	double k2, kx, ky, kz;

 	for (j=0; j<R_MESH; j++) {
 		for (k=0; k<R_MESH; k++) {
			if (j > R_MESH/2.0) {
				kx = 2*PI*double(R_MESH-j)/Lx;
			} else {
				kx = 2*PI*double(j)/Lx;
			}

			if (k > R_MESH/2.0) {
				ky = 2*PI*double(R_MESH-k)/Ly;
			} else {
				ky = 2*PI*double(k)/Ly;
			}

			k2 = kx*kx + ky*ky;

			x  = k*R_MESH + j;
			if (x == 0) {
				g[x] = 1.0;
			} else {
				g[x] = 2.0*(exp(-k2) + k2 - 1.0) / k2 / k2;
			}
		}
	}	
	return;
}

//
// pseudospectral_a - Solves the propagator diffusion equation pseudospectrally for species A.
//
void pseudospectral_a (complex<double> *w) {
	int i,j,k,s, idx;
	double kx, ky, kz; 
	complex<double> temp;
	fftw_complex *qf;   // Initial q to be transformed.
	fftw_complex *a;    // Fourier coefficients
	fftw_complex *h;    // Fourier coefficients
	fftw_plan    qplan, hplan; // FFTW plans.

	// FFTW Initialization
	qf    = (fftw_complex*) fftw_malloc(R_PTS*sizeof(fftw_complex));
	h     = (fftw_complex*) fftw_malloc(R_PTS*sizeof(fftw_complex));
	a     = (fftw_complex*) fftw_malloc(R_PTS*sizeof(fftw_complex));
	qplan = fftw_plan_dft(2, NxNyNz, qf, a, FFTW_FORWARD, FFTW_MEASURE);
	hplan = fftw_plan_dft(2, NxNyNz, h, qf, FFTW_BACKWARD, FFTW_MEASURE);

	// Initial conditions q(x, s=0) = 1.0;
	for (j=0; j<R_MESH; j++) {
		for (k=0; k<R_MESH; k++) {
			idx = j*R_MESH + k;
			qA[idx][0] = 1.0;
		}
	}

	// Apply exp(-Ds*w(x)/2) to q. (Step 1 of pseudospectral algorithm)
	for (s=1; s<S_MESH_A; s++) {
		for (j=0; j<R_MESH; j++) {
			for (k=0; k<R_MESH; k++) {
				idx = j*R_MESH+k;
				temp = qA[idx][s-1] * exp(-w[idx]*dsa/2.0);
				qf[idx][0] = real(temp);
				qf[idx][1] = imag(temp);
			}
		}

		// Transform for next step. Now we multiply the a coefficients by exp(-2*pi*b^2*dsa/j^2/3/Lx/Ly).
		fftw_execute(qplan);
		for (j=0; j<R_MESH; j++) {
			for (k=0; k<R_MESH; k++) {
				idx = j*R_MESH+k;
				a[idx][0] *= 1.0/double(R_PTS);
				a[idx][1] *= 1.0/double(R_PTS);
			}
		}

		for (j=0; j<R_MESH; j++) {
			for (k=0; k<R_MESH; k++) {
				idx = j*R_MESH+k;
				if (j >= R_MESH/2.0) {
					kx = double(R_MESH-j)/Lx;
				} else{
					kx = double(j)/Lx;
				}

				if (k >= R_MESH/2.0) {
					ky = double(R_MESH-k)/Ly;
				} else {
					ky = double(k)/Ly;
				}

				temp      = (a[idx][0] + I*a[idx][1])* exp(-4.0*PI*PI*(kx*kx+ky*ky)*dsa);
				h[idx][0] = real(temp);
				h[idx][1] = imag(temp);
			}
		}

		// Do inverse Fourier transform to prepare for final operator.
		fftw_execute(hplan);

		for (j=0; j<R_MESH; j++) {
			for (k=0; k<R_MESH; k++) {
				idx  = j*R_MESH+k;
				temp =  (qf[idx][0] + I*qf[idx][1]) * exp(-w[idx]*dsa/2.0);
				qA[idx][s] = temp;
			}
		}

	}	

	QA = rectangle_integrate_qA()/double(V);

	cout << "QA: " << QA << endl;

	fftw_destroy_plan(qplan);
	fftw_destroy_plan(hplan);
	fftw_free(qf);
	fftw_free(h);
	fftw_free(a);
	return;
}


//
// pseudospectral_b - Solves the propagator diffusion equation pseudospectrally for species B.
//
void pseudospectral_b (complex<double> *w) {
	int i,j,k,s,idx;
	double kx, ky,kz; 
	complex<double> temp;
	fftw_complex *qf;   // Initial q to be transformed.
	fftw_complex *a;    // Fourier coefficients
	fftw_complex *h;    // Fourier coefficients
	fftw_plan    qplan, hplan; // FFTW plans.

	// FFTW Initialization
	qf    = (fftw_complex*) fftw_malloc(R_PTS*sizeof(fftw_complex));
	h     = (fftw_complex*) fftw_malloc(R_PTS*sizeof(fftw_complex));
	a     = (fftw_complex*) fftw_malloc(R_PTS*sizeof(fftw_complex));
	qplan = fftw_plan_dft(2, NxNyNz, qf, a, FFTW_FORWARD, FFTW_MEASURE);
	hplan = fftw_plan_dft(2, NxNyNz, h, qf, FFTW_BACKWARD, FFTW_MEASURE);

	// Initial conditions q(x, s=0) = 1.0;
	for (j=0; j<R_MESH; j++) {
		for (k=0; k<R_MESH; k++) {
			idx = j*R_MESH + k;
			qB[idx][0] = 1.0;
		}
	}

	// Apply exp(-Ds*w(x)/2) to q. (Step 1 of pseudospectral algorithm)
	for (s=1; s<S_MESH_B; s++) {
		for (j=0; j<R_MESH; j++) {
			for (k=0; k<R_MESH; k++) {
				idx = j*R_MESH+k;
				temp = qB[idx][s-1] * exp(-w[idx]*dsb/2.0);
				qf[idx][0] = real(temp);
				qf[idx][1] = imag(temp);
			}
		}

		// Transform for next step. Now we multiply the a coefficients by exp(-2*pi*b^2*dsb/j^2/3/Lx/Ly).
		fftw_execute(qplan);
		for (j=0; j<R_MESH; j++) {
			for (k=0; k<R_MESH; k++) {
				idx = j*R_MESH+k;
				a[idx][0] *= 1.0/double(R_PTS);
				a[idx][1] *= 1.0/double(R_PTS);
			}
		}

		for (j=0; j<R_MESH; j++) {
			for (k=0; k<R_MESH; k++) {
				idx = j*R_MESH+k;
				if (j >= R_MESH/2.0) {
					kx = double(R_MESH-j)/Lx;
				} else {
					kx = double(j)/Lx;
				}

				if (k >= R_MESH/2.0) {
					ky = double(R_MESH-k)/Ly;
				} else {
					ky = double(k)/Ly;
				}

				temp      = (a[idx][0] + I*a[idx][1])* exp(-4.0*PI*PI*(kx*kx+ky*ky)*dsb);
				h[idx][0] = real(temp);
				h[idx][1] = imag(temp);
			}
		}

		// Do inverse Fourier transform to prepare for final operator.
		fftw_execute(hplan);

		for (j=0; j<R_MESH; j++) {
			for (k=0; k<R_MESH; k++) {
				idx  = j*R_MESH+k;
				temp =  (qf[idx][0] + I*qf[idx][1]) * exp(-w[idx]*dsb/2.0);
				qB[idx][s] = temp;
			}	
		}	

	}	

	QB = rectangle_integrate_qB()/double(V);

	fftw_destroy_plan(qplan);
	fftw_destroy_plan(hplan);
	fftw_free(qf);
	fftw_free(h);
	fftw_free(a);
	return;
}


//
// rectangle_integrate_qA - Integrates qA(r,N) using a series of rectangles.
// 
complex<double> rectangle_integrate_qA (void) {
	int i,j;
	complex<double> integral = complex<double>(0.0, 0.0);

	for (i=0; i<R_PTS; i++) {
		integral += qA[i][S_MESH_A-1];
	}

	integral = integral*V/double(R_PTS);

	return integral;
}


//
// rectangle_integrate_qB - Integrates qB(r,N) using a series of rectangles.
// 
complex<double> rectangle_integrate_qB (void) {
	int i,j;
	complex<double> integral = complex<double>(0.0, 0.0);

	for (i=0; i<R_PTS; i++) {
		integral += qB[i][S_MESH_B-1];
	}

	integral = integral*V/double(R_PTS);

	return integral;
}


//
// simpson_integrate - Integrates array using Simpson's Method. h is the subinterval size, equal to 1/(number of points).
// 
complex<double> simpson_integrate (int arr_size, complex<double>*array, double h) {
	int i,j;
	complex<double> integral;

	integral = 3.0/8.0*array[0] + 7.0/6.0*array[1] + 23.0/24.0*array[2];

	for (i=3; i<arr_size - 3; i++) {
		integral += array[i];
	}
	
	integral += 23.0/24.0*array[arr_size-3] + 7.0/6.0*array[arr_size-2] + 3.0/8.0*array[arr_size-1];
	integral *= h;	

	return integral;
}


//
// update_fields - Updates the W+ and W- fields using a semi-implicit Siedel algorithm (SIS).
//
void update_fields (void) {
	int i, j, k, idx;
	complex<double> mean_wp, mean_wx;
	complex<double> pA_k[R_PTS], 
		        pB_k[R_PTS];
	complex<double> gA[R_PTS], gB[R_PTS];
	complex<double> phiT[R_PTS];

	fftw_complex *pA_real;     
	fftw_complex *pB_real;     
	fftw_complex *pA_fourier;  
	fftw_complex *pB_fourier;  
	fftw_complex *wp_real;     
	fftw_complex *wx_real;     
	fftw_complex *wp_fourier;  
	fftw_complex *wx_fourier;  
	fftw_complex *phiT_real;   
	fftw_complex *phiT_fourier;
	fftw_plan    pA_forward, pB_forward, wp_forward, wp_backward, wx_forward, wx_backward, phiT_forward;

	// FFTW Initialization
	pA_real       = (fftw_complex*) fftw_malloc(R_PTS*sizeof(fftw_complex));
	pB_real       = (fftw_complex*) fftw_malloc(R_PTS*sizeof(fftw_complex));
	pA_fourier    = (fftw_complex*) fftw_malloc(R_PTS*sizeof(fftw_complex));
	pB_fourier    = (fftw_complex*) fftw_malloc(R_PTS*sizeof(fftw_complex));
	wp_real       = (fftw_complex*) fftw_malloc(R_PTS*sizeof(fftw_complex));
	wx_real       = (fftw_complex*) fftw_malloc(R_PTS*sizeof(fftw_complex));
	wp_fourier    = (fftw_complex*) fftw_malloc(R_PTS*sizeof(fftw_complex));
	wx_fourier    = (fftw_complex*) fftw_malloc(R_PTS*sizeof(fftw_complex));
	phiT_real     = (fftw_complex*) fftw_malloc(R_PTS*sizeof(fftw_complex));
	phiT_fourier  = (fftw_complex*) fftw_malloc(R_PTS*sizeof(fftw_complex));

	// FFTW plans for all of our necessary transforms.
	pA_forward   = fftw_plan_dft(2, NxNyNz, pA_real, pA_fourier, FFTW_FORWARD, FFTW_MEASURE);
	pB_forward   = fftw_plan_dft(2, NxNyNz, pB_real, pB_fourier, FFTW_FORWARD, FFTW_MEASURE);
	wp_forward   = fftw_plan_dft(2, NxNyNz, wp_real, wp_fourier, FFTW_FORWARD, FFTW_MEASURE);
	wx_forward   = fftw_plan_dft(2, NxNyNz, wx_real, wx_fourier, FFTW_FORWARD, FFTW_MEASURE);
	wp_backward  = fftw_plan_dft(2, NxNyNz, wp_fourier, wp_real, FFTW_BACKWARD, FFTW_MEASURE);
	wx_backward  = fftw_plan_dft(2, NxNyNz, wx_fourier, wx_real, FFTW_BACKWARD, FFTW_MEASURE);
	phiT_forward = fftw_plan_dft(2, NxNyNz, phiT_real, phiT_fourier, FFTW_FORWARD, FFTW_MEASURE);

	// Get the Debye functions.
	debye_gA(gA);
	debye_gB(gB);

	// Transform density fields and put them back into a nice array.
	for (i=0; i<R_PTS; i++) {
	   pA_real[i][0]   = real(pA[i]);
	   pA_real[i][1]   = 0.0;
	   pB_real[i][0]   = real(pB[i]);
	   pB_real[i][1]   = 0.0;
	   phiT_real[i][0] = 1.00;
	   phiT_real[i][1] = 0.00;
	}
	fftw_execute(pA_forward);
	fftw_execute(pB_forward);
	fftw_execute(phiT_forward);
	for (i=0; i<R_PTS; i++) {
		pA_k[i] = (pA_fourier[i][0] + I*pA_fourier[i][1]);
		pB_k[i] = (pB_fourier[i][0] + I*pB_fourier[i][1]);
		phiT[i] = (phiT_fourier[i][0]+ I*phiT_fourier[i][1]);
	}

	// Update W+
	for (i=0; i<R_PTS; i++) {
		wp_real[i][0] = real(wp[i]);
		wp_real[i][1] = imag(wp[i]);
	}
	fftw_execute(wp_forward);
	for (i=0; i<R_PTS; i++) {
		wp[i] = wp_fourier[i][0] + I*wp_fourier[i][1];
		wp[i] = wp[i] + lambda_p*dt*( (pA_k[i] + pB_k[i] - phiT[i])  + (phiA_total*gA[i] + (1.00-phiA_total)*gB[i])*wp[i]);
		wp[i] = wp[i] / (1.00 + lambda_p*dt*(phiA_total*gA[i] + (1.00-phiA_total)*gB[i]));
		wp_fourier[i][0] = real(wp[i]);
		wp_fourier[i][1] = imag(wp[i]);
	}
	wp_fourier[0][0] = 0.0;
	wp_fourier[0][1] = 0.0;
	fftw_execute(wp_backward);
	for (i=0; i<R_PTS; i++) {
		wp[i] = (wp_real[i][0] + I*wp_real[i][1])/double(R_PTS);
	}

	// We require a re-evaluation of the density functions, etc. for the SIS algorithm.
	calc_a_density();
	calc_b_density();

	// Transform density fields and put them back into a nice array.
	for (i=0; i<R_PTS; i++) {
		pA_real[i][0] = real(pA[i]);
		pA_real[i][1] = 0.0;
		pB_real[i][0] = real(pB[i]);
		pB_real[i][1] = 0.0;
	}

	fftw_execute(pA_forward);
	fftw_execute(pB_forward);
	for (i=0; i<R_PTS; i++) {
		pA_k[i] = (pA_fourier[i][0] + I*pA_fourier[i][1]);
		pB_k[i] = (pB_fourier[i][0] + I*pB_fourier[i][1]);
	}


	// Update W-
	for (i=0; i<R_PTS; i++) {
		wx_real[i][0] = real(wx[i]);
		wx_real[i][1] = imag(wx[i]);
	}
	fftw_execute(wx_forward);
	for (i=0; i<R_PTS; i++) {
		wx[i] = (wx_fourier[i][0] + I*wx_fourier[i][1]);
		wx[i] = (wx[i] - lambda_x*dt*(pB_k[i] - pA_k[i]));
		wx[i] = wx[i] / (1.00 + lambda_x*dt*2.0/chiAB);
		wx_fourier[i][0] = real(wx[i]);
		wx_fourier[i][1] = imag(wx[i]);
	}
	wx_fourier[0][0] = 0.0;
	wx_fourier[0][1] = 0.0;
	fftw_execute(wx_backward);
	for (i=0; i<R_PTS; i++) {
		wx[i] = (wx_real[i][0] + I*wx_real[i][1])/double(R_PTS);
	}


	// Clean up clean up everybody everywhere clean up clean up everybody do your share.
	fftw_destroy_plan(pA_forward);
	fftw_destroy_plan(pB_forward);
	fftw_destroy_plan(wp_forward);
	fftw_destroy_plan(wx_forward);
	fftw_destroy_plan(wp_backward);
	fftw_destroy_plan(wx_backward);
	fftw_destroy_plan(phiT_forward);
	fftw_free(pA_real);
	fftw_free(pB_real);
	fftw_free(pA_fourier);
	fftw_free(pB_fourier);
	fftw_free(wx_real);
	fftw_free(wp_real);
	fftw_free(wx_fourier);
	fftw_free(wp_fourier);
	fftw_free(phiT_real);
	fftw_free(phiT_fourier);
	return;
}


//
// write_out - Writes fields, densities, etc. to disk.
//
void write_out (int iteration) {
	FILE *pA_out, *pB_out;
	FILE *wp_out, *wx_out;
	FILE *ga_out, *gb_out;
	FILE *density;
	int i,j,k,idx;

	pA_out = fopen("pA.dat", "w");
	pB_out = fopen("pB.dat", "w");
	wp_out = fopen("wp.dat", "w");
	wx_out = fopen("wx.dat", "w");

	for (j=0; j<R_MESH; j++) {
		for (k=0; k<R_MESH; k++) {
			idx = i*R_MESH*R_MESH + j*R_MESH + k;
			fprintf(pA_out, "%lf %lf %lf\n", j*Lx/R_MESH, k*Ly/R_MESH, real(pA[idx]));
			fprintf(pB_out, "%lf %lf %lf\n", j*Lx/R_MESH, k*Ly/R_MESH, real(pB[idx]));
			fprintf(wp_out, "%lf %lf %lf\n", j*Lx/R_MESH, k*Ly/R_MESH, real(wp[idx]));
			fprintf(wx_out, "%lf %lf %lf\n", j*Lx/R_MESH, k*Ly/R_MESH, real(wx[idx]));
		}
	}

	fclose(pA_out);
	fclose(pB_out);
	fclose(wp_out);
	fclose(wx_out);
	return;
}
