#include "GMRes.h"

using namespace std;

GMRes::GMRes(int krylovDemension, int mRestart, int unknows, double tolerance)
		: krylovDemension(krylovDemension), mRestart(mRestart), unknows(unknows), tolerance(tolerance) {
	v = new complex<double>[unknows];
	res = new complex<double>[unknows];
	Q = new complex<double>[(krylovDemension + 1) * unknows];
	H = new complex<double>[(krylovDemension + 1) * krylovDemension];
	omega = new complex<double>[(krylovDemension + 1) * (krylovDemension + 1)];
}

GMRes::~GMRes() {
	auto FREE = [](complex<double> *&p) {
		if (p != nullptr) {
			delete p;
			p = nullptr;
		}
	};
	auto FREE2D = [](complex<double> *&p) {
		if (p != nullptr) {
			delete[] p;
			p = nullptr;
		}
	};
	FREE2D(v);
	FREE2D(res);
	FREE2D(Q);
	FREE2D(H);
	FREE2D(omega);
}

double GMRes::CheckCovergence(complex<double> *x, complex<double> *A, complex<double> *b) {
	complex<double> *residual = new complex<double>[unknows];
	cblas_zgemv(CblasRowMajor, CblasNoTrans, unknows, unknows, &ONE, A, unknows, x, 1, &ZERO, residual, 1);
	cblas_zaxpy(unknows, &NEGONE, b, 1, residual, 1);
	double norm_b = cblas_dznrm2(unknows, b, 1);
	double norm_r = cblas_dznrm2(unknows, residual, 1);
	delete[] residual;
	cout << norm_r / norm_b << endl;
	return norm_r / norm_b;
}

int GMRes::Solve(complex<double> *A, complex<double> *x0, complex<double> *b) {
	double error = RestartGMRes(A, x0, b);
	return 1;
}

double GMRes::RestartGMRes(complex<double> *A, complex<double> *x0, complex<double> *b) {
	double error = 1;
	int outloops = 0, iter;
	while (outloops < mRestart && error > tolerance) {
		error = InnerLoop(A, x0, b);
	}
	return error;
}

double GMRes::InnerLoop(complex<double> *A, complex<double> *x0, complex<double> *b) {
	complex<double> *residual = new complex<double>[unknows];
	complex<double> *Q = new complex<double>[unknows * (krylovDemension + 1)];
	complex<double> *H = new complex<double>[(krylovDemension + 1) * krylovDemension];
	complex<double> *v = new complex<double>[unknows];
	complex<double> *q = new complex<double>[unknows];

	cblas_zgemv(CblasRowMajor, CblasNoTrans, unknows, unknows, &ONE, A, unknows, x0, 1, &ZERO, residual, 1);
	cblas_zaxpby(unknows, &ONE, b, 1, &NEGONE, residual, 1);

	complex<double> one_norm = ONE / cblas_dznrm2(unknows, residual, 1);
	cblas_zaxpy(unknows, &one_norm, residual, 1, q, 1);
	cblas_zcopy(unknows, q, 1, Q, krylovDemension + 1);

	for (int k = 0; k < krylovDemension; k++) {
		cblas_zgemv(CblasRowMajor, CblasNoTrans, unknows, unknows, &ONE, A, unknows, q, 1, &ZERO, v, 1);
		for (int j = 0; j < k + 1; j++) {
			cblas_zdotu_sub(unknows, Q + j, krylovDemension + 1, v, 1,
			                &H[j * krylovDemension + k]);
			complex<double> coe = -H[j * krylovDemension + k];
			cblas_zaxpy(unknows, &coe, Q + j, krylovDemension + 1, v, 1);
		}
		H[(k + 1) * krylovDemension + k] = cblas_dznrm2(unknows, v, 1);
		complex<double> coe = ONE / H[(k + 1) * krylovDemension + k];
		cblas_zaxpby(unknows, &coe, v, 1, &ZERO, q, 1);
		cblas_zcopy(unknows, q, 1, Q + k + 1, krylovDemension + 1);

	}

}

void GMRes::LeastSquare(complex<double> *y, int step) {
	//	 lapack_int LAPACKE_zgeqrf(int matrix_layout, lapack_int m, lapack_int n, lapack_complex_double* a, lapack_int
	// lda, lapack_complex_double* tau);
	// LAPACKE_zgeqrf(LAPACK_ROW_MAJOR,step+2,step+1,A,krylovDemension,tau);
	complex<double> *A = new complex<double>[(step + 2) * (step + 1)];
	for (int i = 0; i < step + 2; i++) {
		for (int j = 0; j < step + 1; j++) {
			A[i * (step + 1) + j] = H[i * krylovDemension + j];
		}
	}
	int ret = LAPACKE_zgels(LAPACK_ROW_MAJOR, 'N', step + 2, step + 1, 1, A, step + 1, y, 1);
	delete[] A;
	if (ret < 0) {
		cout << "the " << -ret << "-th parameter had an illegal value.";
	} else if (ret > 0) {
		cout << "the " << ret
		     << "-th diagonal element of the triangular factor of A is zero, so that A does not have full rank; the "
				     "least squares solution could not be computed."
		     << endl;
	}
}
