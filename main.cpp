#include "GMRes.h"
#include <iostream>
using namespace std;

int main()
{

	std::complex<double> M[2];
	M[0] = complex<double>(-1, 0);
	M[1] = complex<double>(-2, 1);
	//	LAPacke_zgeqr
	int unknowns = 100;
	int krylov = 100;
	double error = 0.01;

	GMRes solver(krylov, 1, unknowns, error);
	complex<double> *Q = new complex<double>[11000];
	complex<double> *H = new complex<double>[11000];
	complex<double> *r0 = new complex<double>[100];
	complex<double> *A = new complex<double>[10000];
	for (size_t i = 0; i < 100; i++)
		r0[i] = i + 1;
	for (int i = 0; i < 100; i++) {
		for (int j = 0; j < 100; j++) {
			if (i == j)
				A[i * 100 + j] = i;
			else
				A[i * 100 + j] = 0;
		}
	}
	complex<double> *x0 = new complex<double>[100];
	solver.Solve(A, x0, r0);
	complex<double> ONE = 1.0;
	complex<double> ZERO = 0.0;
	complex<double> *b = new complex<double>[unknowns];
	cblas_zgemv(CblasRowMajor, CblasNoTrans, unknowns, unknowns, &ONE, A, unknowns, x0, 1, &ZERO, b, 1);
	getchar();
}