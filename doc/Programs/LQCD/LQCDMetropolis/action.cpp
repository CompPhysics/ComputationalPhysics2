#include "action.h"

Action::Action(int NPathPoints, double new_a)
{
    N = NPathPoints;
    a = new_a;
}

double Action::getAction(double * x, int i)
{
    int i_prev = (i-1) % N; // mod boundary conditions
    int i_next = (i+1) % N; // mod boundary conditions
    return x[i]*(x[i] - x[i_prev] - x[i_next])/a + a*potential(x[i]);
}
