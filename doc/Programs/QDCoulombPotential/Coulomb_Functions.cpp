#include "Coulomb_Functions.hpp"

double Coulomb_HO(double &hw, int &ni, int &mi, int &nj, int &mj, int &nk, int &mk, int &nl, int &ml)
{
  int g1, g2, g3, g4, G, L;
  double dir = 0.0;
  double exch = 0.0;
  double LogRatio1;
  double LogProd2;
  double LogRatio2;
  double temp;
  if(mi + mj != mk + ml){ return 0.0; }
  for(int j1 = 0; j1 <= ni; ++j1){
    for(int j2 = 0; j2 <= nj; ++j2){
      for(int j3 = 0; j3 <= nl; ++j3){
	for(int j4 = 0; j4 <= nk; ++j4){
	  g1 = int(j1 + j4 + 0.5*(std::abs(mi) + mi) + 0.5*(std::abs(mk) - mk));
	  g2 = int(j2 + j3 + 0.5*(std::abs(mj) + mj) + 0.5*(std::abs(ml) - ml));
	  g3 = int(j3 + j2 + 0.5*(std::abs(ml) + ml) + 0.5*(std::abs(mj) - mj));
	  g4 = int(j4 + j1 + 0.5*(std::abs(mk) + mk) + 0.5*(std::abs(mi) - mi));
	  G = g1 + g2 + g3 + g4;
	  LogRatio1 = logratio1(j1, j2, j3, j4);
	  LogProd2 = logproduct2(ni, mi, nj, mj, nl, ml, nk, mk, j1, j2, j3, j4);
	  LogRatio2 = logratio2(G);
	  temp = 0.0;
	  for(int l1 = 0; l1 <= g1; ++l1){
	    for(int l2 = 0; l2 <= g2; ++l2){
	      for(int l3 = 0; l3 <= g3; ++l3){
		for(int l4 = 0; l4 <= g4; ++l4){
		  if(l1 + l2 != l3 + l4){ continue; }
		  L = l1 + l2 + l3 + l4;
		  temp += (-2*((g2 + g3 - l2 - l3)%2) + 1) * std::exp(logproduct3(l1, l2, l3, l4, g1, g2, g3, g4) + std::lgamma(1.0 + 0.5*L) + std::lgamma(0.5*(G - L + 1.0)));
		}
	      }
	    }
	  }
	  dir += (-2*((j1 + j2 + j3 + j4)%2) + 1) * std::exp(LogRatio1 + LogProd2 + LogRatio2) * temp;
	}
      }
    }
  }
  dir *= product1(ni, mi, nj, mj, nl, ml, nk, mk);

  /*
  for(int j1 = 0; j1 <= ni; ++j1){
    for(int j2 = 0; j2 <= nj; ++j2){
      for(int j3 = 0; j3 <= nk; ++j3){
	for(int j4 = 0; j4 <= nl; ++j4){
	  g1 = int(j1 + j4 + 0.5*(std::abs(mi) + mi) + 0.5*(std::abs(ml) - ml));
	  g2 = int(j2 + j3 + 0.5*(std::abs(mj) + mj) + 0.5*(std::abs(mk) - mk));
	  g3 = int(j3 + j2 + 0.5*(std::abs(mk) + mk) + 0.5*(std::abs(mj) - mj));
	  g4 = int(j4 + j1 + 0.5*(std::abs(ml) + ml) + 0.5*(std::abs(mi) - mi));
	  G = g1 + g2 + g3 + g4;
	  LogRatio1 = logratio1(j1, j2, j3, j4);
	  LogProd2 = logproduct2(ni, mi, nj, mj, nk, mk, nl, ml, j1, j2, j3, j4);
	  LogRatio2 = logratio2(G);
	  temp = 0.0;
	  for(int l1 = 0; l1 <= g1; ++l1){
	    for(int l2 = 0; l2 <= g2; ++l2){
	      for(int l3 = 0; l3 <= g3; ++l3){
		for(int l4 = 0; l4 <= g4; ++l4){
		  if(l1 + l2 != l3 + l4){ continue; }
		  L = l1 + l2 + l3 + l4;
		  temp += (-2*((g2 + g3 - l2 - l3)%2) + 1) * std::exp(logproduct3(l1, l2, l3, l4, g1, g2, g3, g4) + std::lgamma(1.0 + 0.5*L) + std::lgamma(0.5*(G - L + 1.0)));
		}
	      }
	    }
	  }
	  exch += (-2*((j1 + j2 + j3 + j4)%2) + 1) * std::exp(LogRatio1 + LogProd2 + LogRatio2) * temp;
	}
      }
    }
  }

  exch *= product1(ni, mi, nj, mj, nk, mk, nl, ml);
  */
  return std::sqrt(hw)*(dir - exch);
}

double logfac(int &n)
{
  if(n < 0){ std::cerr << n << " : LogFactorial intput should be >= 0" << std::endl; exit(1); }
  double fac = 0.0;
  for(int a = 2; a < n+1; a++){ fac += std::log(a); }
  return fac;
}

double logratio1(int &int1, int &int2, int &int3, int &int4)
{
  return -logfac(int1) - logfac(int2) - logfac(int3) - logfac(int4);
}

double logratio2(int &G)
{
  return -0.5 * (G + 1) * log(2);
}

double product1(int &n1, int &m1, int &n2, int &m2, int &n3, int &m3, int &n4, int &m4)
{
  double prod = logfac(n1) + logfac(n2) + logfac(n3) + logfac(n4);
  int arg1 = n1 + std::abs(m1);
  int arg2 = n2 + std::abs(m2);
  int arg3 = n3 + std::abs(m3);
  int arg4 = n4 + std::abs(m4);
  prod -= (logfac(arg1) + logfac(arg2) + logfac(arg3) + logfac(arg4));
  prod *= 0.5;
  return std::exp(prod);
}

double logproduct2(int &n1, int &m1, int &n2, int &m2, int &n3, int &m3, int &n4, int &m4, int &j1, int &j2, int &j3, int &j4)
{
  int arg1 = n1 + std::abs(m1);
  int arg2 = n2 + std::abs(m2);
  int arg3 = n3 + std::abs(m3);
  int arg4 = n4 + std::abs(m4);
  int narg1 = n1 - j1;
  int narg2 = n2 - j2;
  int narg3 = n3 - j3;
  int narg4 = n4 - j4;
  int jarg1 = j1 + std::abs(m1);
  int jarg2 = j2 + std::abs(m2);
  int jarg3 = j3 + std::abs(m3);
  int jarg4 = j4 + std::abs(m4);
  double prod = logfac(arg1) + logfac(arg2) + logfac(arg3) + logfac(arg4);
  prod -= (logfac(narg1) + logfac(narg2) + logfac(narg3) + logfac(narg4));
  prod -= (logfac(jarg1) + logfac(jarg2) + logfac(jarg3) + logfac(jarg4));
  return prod;
}

double logproduct3(int &l1, int &l2, int &l3, int &l4, int &g1, int &g2, int &g3, int &g4)
{
  int garg1 = g1 - l1;
  int garg2 = g2 - l2;
  int garg3 = g3 - l3;
  int garg4 = g4 - l4;
  double prod = logfac(g1) + logfac(g2) + logfac(g3) + logfac(g4);
  prod -= (logfac(l1) + logfac(l2) + logfac(l3) + logfac(l4));
  prod -= (logfac(garg1) + logfac(garg2) + logfac(garg3) + logfac(garg4));
  return prod;
}
