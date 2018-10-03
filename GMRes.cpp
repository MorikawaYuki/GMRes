#include "GMRes.h"

using namespace std;

GMRes::GMRes(int krylovDemension, int mRestart, int unknows, double tolerance)
		: krylovDemension(krylovDemension), mRestart(mRestart), unknows(unknows), tolerance(tolerance) {
	res_n = new complex<double>[unknows];
	residual = new complex<double>[unknows];
	Q = new complex<double>[unknows * (krylovDemension + 1)];
	H = new complex<double>[(krylovDemension + 1) * krylovDemension];
	v = new complex<double>[unknows];
	q = new complex<double>[unknows];
	y = new complex<double>[krylovDemension + 1];
	x = new complex<double>[unknows];
	beta = new complex<double>[krylovDemension + 1];
}

GMRes::~GMRes() {
	auto FREE = [](complex<double> *&p) {
		if (p != nullptr) {
			delete p;
			p = nullptr;
		}
	};
	FREE(res_n);
	FREE(residual);
	FREE(Q);
	FREE(H);
	FREE(v);
	FREE(q);
	FREE(y);
	FREE(x);
	FREE(beta);


}

double GMRes::CheckCovergence(complex<double> *x, complex<double> *A, complex<double> *b) {
	
	cblas_zcopy(unknows, b, 1, res_n, 1);
	cblas_zgemv(CblasRowMajor, CblasNoTrans, unknows, unknows, &ONE, A, unknows, x, 1, &NEGONE, res_n, 1);
	
	double error = cblas_dznrm2(unknows, res_n, 1)/norm_b;
	
	return error;
}

int GMRes::Solve(complex<double> *A, complex<double> *x0, complex<double> *b) {
	double error = RestartGMRes(A, x0, b);
	return 1;
}

double GMRes::RestartGMRes(complex<double> *A, complex<double> *x0, complex<double> *b) {
	double error = 1;
	int outloops = 0;
	while (outloops < mRestart && error > tolerance) {
		outloops++;
		error = InnerLoop(A, x0, b);
		cout << "outloop:" << outloops << "  error:" << error << endl;
	}
	return error;
}

double GMRes::InnerLoop(complex<double> *A, complex<double> *x0, complex<double> *b) {



	double error = 1;

	norm_b = cblas_dznrm2(unknows, b, 1);
	cblas_zgemv(CblasRowMajor, CblasNoTrans, unknows, unknows, &ONE, A, unknows, x0, 1, &ZERO, residual, 1);
	cblas_zaxpby(unknows, &ONE, b, 1, &NEGONE, residual, 1);

	norm_residual = cblas_dznrm2(unknows, residual, 1);
	beta[0] = norm_residual;
	complex<double> one_norm = ONE / norm_residual;


	cblas_zaxpby(unknows, &one_norm, residual, 1, &ZERO, q, 1);
	cblas_zcopy(unknows, q, 1, Q, krylovDemension + 1);

	for (int k = 0; k < krylovDemension; k++) {
		cblas_zgemv(CblasRowMajor, CblasNoTrans, unknows, unknows, &ONE, A, unknows, q, 1, &ZERO, v, 1);
		for (int j = 0; j < k + 1; j++) {
			cblas_zdotc_sub(unknows, Q + j, krylovDemension + 1, v, 1,
			                &H[j * krylovDemension + k]);
			
			complex<double> coe = NEGONE * H[j * krylovDemension + k];
			cblas_zaxpy(unknows, &coe, Q + j, krylovDemension + 1, v, 1);
		}
		H[(k + 1) * krylovDemension + k] = cblas_dznrm2(unknows, v, 1);
		
		double tmp = abs(H[(k + 1) * krylovDemension + k]) / norm_b;
	

		complex<double> coe = ONE / H[(k + 1) * krylovDemension + k];
		cblas_zaxpby(unknows, &coe, v, 1, &ZERO, q, 1);
		cblas_zcopy(unknows, q, 1, Q + k + 1, krylovDemension + 1);
		

		if (k == krylovDemension - 1)
		{
			cblas_zcopy(k + 2, beta, 1, y, 1);
			LeastSquare(y, k, H);

			cblas_zgemv(CblasRowMajor, CblasNoTrans, unknows, k + 1, &ONE, Q, krylovDemension + 1, y, 1, &ZERO, x, 1);
			cblas_zaxpy(unknows, &ONE, x0, 1, x, 1);

			error = CheckCovergence(x, A, b);
			cout << error << endl;
			if (error < tolerance)
			{
				break;
			}
		}

		
	
	}

	cblas_zcopy(unknows, x, 1, x0, 1);

	return error;
}

void GMRes::LeastSquare(complex<double> *y, int step, complex<double> * H) {
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
