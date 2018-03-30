#ifndef ACTION_H
#define ACTION_H


class Action
{
private:
    double (*potential)(double x);
    double a;
    int N;
public:
    Action(int NPathPoints, double new_a);
    double getAction(double * x, int i);

    void setPotential(double (*pot)(double x)) { potential = pot; }
};

#endif // ACTION_H
