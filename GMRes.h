#ifndef GMRES_H
#define GMRES_H

#define MKL_Complex16 std::complex<double>

#include <complex>
#include <iostream>
#include <mkl.h>
using namespace std;

class GMRes
{
  public:
	const int krylovDemension;
	const int mRestart;
	const int unknows;
	const double tolerance;


	double norm_b,norm_residual;

	const complex<double> ONE = 1.0;
	const complex<double> NEGONE = -1.0;
	const complex<double> ZERO = 0.0;
	complex<double> *residual, *Q, *H, *v, *q, *y, *x, *beta, *res_n;


	GMRes(int krylovDemension, int mRestart, int unknows, double tolerance);
	~GMRes();
	int Solve(complex<double> *A, complex<double> *x0, complex<double> *b);
	double RestartGMRes(complex<double> *A, complex<double> *x0, complex<double> *b);
	double InnerLoop(complex<double> *A, complex<double> *x0, complex<double> *b);
	void LeastSquare(complex<double> *y, int step, complex<double> * H);
	double CheckCovergence(complex<double> *x, complex<double> *A, complex<double> *b);
};

#endif // GMRES_H