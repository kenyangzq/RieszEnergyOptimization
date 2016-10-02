#ifndef MIN_H    
#define MIN_H

// TODO add multiprecision arithmetic with either https://gmplib.org/, or http://www.mpfr.org/
// TODO use an autodifferentiation library like CppAD, http://www.coin-or.org/CppAD/
//
//
//


#include <iomanip>
#include <fstream>
#include <Eigen/Dense>
#include <iostream>
#include "meta.h"
#include "problem.h"
#include "solver/lbfgssolver.h"
#include "solver/lbfgsbsolver.h"

using namespace std;

#define PI 3.141592653589793
class minimizeEnergy : public cppoptlib::Problem<double> {
    
    double cutoff_radius;
    double s;
    int cubes_per_side;
    int dim;
    int numpts;
    cppoptlib::Matrix<double> Cubes;
    cppoptlib::Matrix<double> ptsND;
    
    
public:
    minimizeEnergy(double r, double s_value, int d, int n, int c, int c_cube, int max_neighbor)
    :cutoff_radius(r), s(s_value), cubes_per_side(c), dim(d), numpts(n), Cubes(max_neighbor+1, c_cube), ptsND(dim, numpts){};
    using typename cppoptlib::Problem<double>::Scalar;
    using typename cppoptlib::Problem<double>::TVector;
    double TruncatedEnergy(const cppoptlib::Vector<double> & V, const cppoptlib::Vector<int> & neighbors, const int index);
    int Point2Cube(const cppoptlib::Vector<double> & thisPt);
    void FindNeighborCubes(int cube_index, cppoptlib::Vector<int> & neighbors);
    void BuildIndex(const cppoptlib::Vector<double> &);
    double value(const cppoptlib::Vector<double> &x);
	// doubt that cutoff prime or cutoff should change as well
	//
    void gradient(const cppoptlib::Vector<double> &x, cppoptlib::Vector<double> &grad);
};

void openFile (ifstream & inputfile, string name);

string ParseControlFile(ifstream & inputfile, int &dim, int &numpts, double & s, int & c, int & max_neighbor, int & history_size);

void To3D(const cppoptlib::Vector<double> & angles, cppoptlib::Vector<double> & coords);


void ToND(const cppoptlib::Vector<double> & angles, cppoptlib::Vector<double> & coords) ;


void Flatten(const Eigen::MatrixXd & M, cppoptlib::Vector<double> & V );

double cutoff(const double & distance, const double & cut);


double cutoff_prime(const double  & distance, const double  & cutoff);

void ComputeJacobian(const double & theta, const double & phi, cppoptlib::Matrix<double> & temp);

void ToAngle(cppoptlib::Vector<double> & angels, const cppoptlib::Vector<double> & coords) ;

void AngleGradient(const cppoptlib::Vector<double> & all_angles, const int & pt_index, const double & s_power, cppoptlib::Vector<double> & output);

void FullGradient(const cppoptlib::Vector<double> & all_angles, const double & s_power, cppoptlib::Vector<double> & output);

void ComputeJacobianN(const cppoptlib::Vector<double> & angles, cppoptlib::Matrix<double> &temp);

double dist_squared(const cppoptlib::Vector<double> & angles1,const cppoptlib::Vector<double> & angles2);

void ToAngles(cppoptlib::Matrix<double> & all_points, cppoptlib::Matrix<double> & all_angles, int dim);

double Energy(const cppoptlib::Vector<double> & V, const double s_power, const int dim);

void ToAngles(Eigen::MatrixXd & all_points, Eigen::MatrixXd & all_angles);

void writeFile (ofstream & outputfile, string name, cppoptlib::Vector<double> V, int dim);

void randptSphere(double coordinates[], int dim);

#endif
