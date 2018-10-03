#include "GMRes.h"
#include <time.h>
#include <iostream>
#include <fstream>
#include<vector>
using namespace std;

int main()
{
	double dur;
	clock_t start, end;
	

	//	LAPacke_zgeqr


	complex<double> tmp;
	vector<complex<double>> z;
	vector<complex<double>> v;
	ifstream in("D:\\workspace\\GMRes\\z_in.txt");
	while (in >> tmp)
	{
		z.push_back(tmp);
	}
	in.close();
	in.open("D:\\workspace\\GMRes\\v0.txt");
	while (in >> tmp)
	{
		v.push_back(tmp);
	}
	in.close();
	

	int unknowns = v.size();
	int krylov = 100;
	double error = 0.002;
	
	GMRes solver(krylov, 31, unknowns, error);

	complex<double> *x0 = new complex<double>[unknowns];
	complex<double> ONE = 1.0;
	complex<double> ZERO = 0.0;
	complex<double> *b = new complex<double>[unknowns];
	complex<double> *A = new complex<double>[unknowns*unknowns];
	int offset = 0;

	for (int i = 0; i < unknowns; i++)
	{
		b[i] = v[i];
		for (int j = 0; j < unknowns; j++)
		{
			A[offset] = z[offset];
			offset++;
		}
	}
	
	start = clock();

	solver.Solve(A, x0, b);

	end = clock();
	dur = (double)(end - start);

	printf("Use Time:%f\n", (dur / CLOCKS_PER_SEC));


	cblas_zgemv(CblasRowMajor, CblasNoTrans, unknowns, unknowns, &ONE, A, unknowns, x0, 1, &ZERO, b, 1);


	for (int i = 90; i < 100; i++)
		cout << x0[i] << " ";
	cout << endl;
	getchar();
}