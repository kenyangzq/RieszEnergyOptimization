// TODO add multiprecision arithmetic with either https://gmplib.org/, or http://www.mpfr.org/
// TODO use an autodifferentiation library like CppAD, http://www.coin-or.org/CppAD/
//
//
//
#include <fstream>
#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include "meta.h"
#include "problem.h"
#include "solver/lbfgsbsolver.h"
#include <omp.h>
//#include "solver/lbfgssolver.h"
#define PI 3.141592653589793

using namespace std;

// we define a new problem 
// we use a templated-class rather than "auto"-lambda function for a clean architecture



ifstream & openFile (ifstream & inputfile, string name){
    inputfile.open(name.c_str());
    if (inputfile.fail()) {
        cout << "Error opening input data file\n";
        exit(1);
    }
    return inputfile;
}

void ParseControlFile(ifstream & inputfile, int &dim, int &numpts, double & s){

    bool infile = false;
    int numpfiles = 0;
    string filename = "";

    string line;
    int lineNumber = 0;
    while (! inputfile.eof()) {
        lineNumber++;
        getline(inputfile, line);
        stringstream tmp(line);
        string k;
        if (lineNumber == 4) {
            tmp >> k >> k >> s;
        }else if (lineNumber == 5) {
            tmp >> k >> k >> dim;
        }else if (lineNumber == 7) {
            tmp >> k >> k >> infile;
        }else if (lineNumber == 8 && !infile) {
            tmp >> k >> k >> numpts;
        }else if (lineNumber == 10) {
            tmp >> k >> k >> numpfiles;
        }else if (lineNumber == 12) {
            tmp >> k >> k >> filename;
            filename = filename + "final.txt";
        }
    }
    cout << "Summary of the control file:\n\n";
    cout << "Dimension: " << dim << "\n";
    cout << "Infile request: " << infile << "\n";
    if (!infile) cout << "Number of points: " << numpts << "\n";
    cout << "Number of output files: " << numpfiles << "\n";
    cout << "Output filename: " << filename << "\n\n\n\n";

}

double dist_squared(const cppoptlib::Vector<double> & angles1,const cppoptlib::Vector<double> & angles2)
{
    return (2-2*(sin(angles1(1))*sin(angles2(1))*cos(angles1(0)-angles2(0))+cos(angles1(1))*cos(angles2(1))));
}

void To3D(const Eigen::Matrix<double, 1, 2> & angles, Eigen::Matrix<double, 1, 3> & coords)
{
    coords(0) = cos(angles(0)) * sin(angles(1));
    coords(1) = sin(angles(0)) * sin(angles(1));
    coords(2) = cos(angles(1));
}

void ToVector(const Eigen::MatrixXd & M, cppoptlib::Vector<double> & V )
{ 
    int c = M.cols();
    for (int i=0; i<M.rows(); ++i )
        V.segment(i*c,c) = M.row(i);
}

void ComputeJacobian(const double & theta, const double & phi, Eigen::Matrix<double, 3, 2> & temp){
    //x = sin(phi) cos(theta)
    //y = sin(phi) sin(theta)
    //z = cos(phi)
    //
    // derivatives w.r.t. theta:
    temp(0,0) = -sin(phi) * sin(theta); // x
    temp(1,0) =  sin(phi) * cos(theta); // y
    temp(2,0) =  0;                      // z
    // derivatives w.r.t. phi:
    temp(0,1) =  cos(phi) * cos(theta); // x
    temp(1,1) =  cos(phi) * sin(theta); // y
    temp(2,1) = -sin(phi);             // z
}


void AngleGradient(const cppoptlib::Vector<double> & all_angles, const int & pt_index, const double & s_power, cppoptlib::Vector<double> & output)
{
    Eigen::Matrix<double, 1, 3> temp_sum, temp_pt, temp_i, temp;
    Eigen::Matrix<double, 3, 2> temp_jacobian;
    temp_sum.setZero();
    To3D(all_angles.segment<2>(pt_index*2), temp_pt);
#pragma omp for ordered schedule(dynamic) private(temp_i, temp)
    for (int i=0; i<pt_index; ++i)
    {
        To3D(all_angles.segment<2>(i*2), temp_i);
        temp = temp_pt - temp_i;
#pragma omp critical
        {
            temp_sum += pow(temp.dot(temp), -1-s_power/2.0) * temp;
        }
    }
#pragma omp for ordered schedule(dynamic) private(temp_i, temp)
    for (int i=pt_index+1; i<all_angles.rows()/2; ++i)
    {
        To3D(all_angles.segment<2>(i*2), temp_i);
        temp = temp_pt - temp_i;
#pragma omp critical
        {
            temp_sum += pow(temp.dot(temp), -1-s_power/2.0) * temp;
        }
    }
    temp_sum *= -s_power;
    //cout << "here is your sum: "<< temp_sum << endl;
    ComputeJacobian(all_angles(pt_index*2+0), all_angles(pt_index*2+1), temp_jacobian);
    output.segment<2>(pt_index*2) = temp_sum * temp_jacobian;
}

void FullGradient(const cppoptlib::Vector<double> & all_angles, const double & s_power, cppoptlib::Vector<double> & output)
{
#pragma omp for schedule(dynamic) 
    for(int i=0; i<all_angles.size()/2; ++i)
    {
        AngleGradient(all_angles, i, s_power, output);
   //int nThreads=omp_get_num_threads();
        //std::cout<<nThreads<<"\t nThreads"<<std::endl;
    }
}

double EnergyMatrix(const cppoptlib::Matrix<double> & M, const double s_power)
    //computes energy using matrices
{
    // M contains spherical coordinates
    double e = 0;
#pragma omp for ordered schedule(dynamic)
    for (int i=0; i<M.rows(); ++i)
    {
        for (int j=0; j<i; ++j)
        {
            e += pow(dist_squared(M.row(i), M.row(j)), -s_power/2.0);
        }
    }
    return 2.0 * e;
}

double Energy(const cppoptlib::Vector<double> & V, const double s_power, const int dim)
{
    // V contains spherical coordinates
    double e = 0;
#pragma omp for ordered schedule(dynamic)
    for (int i=0; i<V.size()/dim; ++i)
    {
        for (int j=0; j<i; ++j)
        {
#pragma omp atomic
            e += pow(dist_squared(V.segment<2>(i*dim), V.segment<2>(j*dim)), -s_power/2.0);
        }
    }
    return 2.0 * e;
}

void ToAngles(Eigen::MatrixXd & all_points, Eigen::MatrixXd & all_angles) 
{
    //cppoptlib::Matrix<double> angles(numpts, dim-1);
    for (int i = 0; i < all_points.rows(); i++) {
        double x = all_points(i,0);
        double y = all_points(i,1);
        double z = all_points(i,2);
        double r = sqrt(x*x+y*y+z*z);
        all_angles(i,0) = atan2(y,x);
        all_angles(i,1) = acos(z/r);
    }
}

class DemoProblem : public cppoptlib::Problem<double> {
    int dim;
    double s;
    public:
    DemoProblem(int dim_value, double s_value):dim(dim_value),s(s_value){}
    // this is just the objective (NOT optional)
    //double value(const Vector<double> &x) {
    //const double t1 = (1 - x[0]);
    //const double t2 = (x[1] - x[0] * x[0]);
    //return   t1 * t1 + 100 * t2 * t2;
    //}
    //
    double value(const cppoptlib::Vector<double> &x) {
        return Energy(x, s, dim-1);
    }

    // if you calculated the derivative by hand
    // you can implement it here (OPTIONAL)
    // otherwise it will fall back to (bad) numerical finite differences
    void gradient(const cppoptlib::Vector<double> &x, cppoptlib::Vector<double> &grad) {
        FullGradient(x, s, grad);
    }
};
/////////////////////////////////////////////////////////////////////////////////
int main(int argc, char const *argv[]) {

    double s;
    int dim = 0, numpts=0;
    ifstream inputfile, pointfile;
    openFile(inputfile, "control.inp");
    ParseControlFile(inputfile, dim, numpts, s);
    inputfile.close();
    openFile(pointfile, "spherefinal.txt");
    Eigen::MatrixXd X(numpts, dim), A(numpts, dim-1);
    // read points
    int lineNumber = 0;
    while (!pointfile.eof() && lineNumber < numpts)
    {
        for (int i=0; i<3;  ++i)
        {
            pointfile >> X(lineNumber, i);
        }
        lineNumber++;
    }
    pointfile.close();
    //
    //
    ToAngles(X,A);

    //Eigen::Matrix<double, 1,3> v;
    cppoptlib::Vector<double> V(A.size()), G(A.size());
    ToVector(A,V);

    //initialize the DemoProblem-problem
    //
    //
    for (int i = 0; i< 100; ++i)
        FullGradient(V, s, G);
    //cout << "  original energy: " << Energy(V, s, 2) << endl;
    //DemoProblem f(dim, s);

    // bounds for lbfgsb
    //cppoptlib::Vector<double> lowerBound(V.size());
    //lowerBound.setZero();
    //for (int i=0; i<V.size()/2; ++i)
    //{
    //lowerBound(2*i) = -PI;
    //}
    //cppoptlib::Vector<double> upperBound = cppoptlib::Vector<double>::Ones(V.size())*PI;
    //f.setLowerBound(lowerBound);
    //f.setUpperBound(upperBound);
    //
    //
    // first check the given derivative 
    //bool probably_correct = f.checkGradient(V);
    //cout << probably_correct << endl;

    //cppoptlib::LbfgsbSolver<double> solver;
    //cppoptlib::LbfgsbSolver<double> solver;
    //solver.minimize(f, V);
    ////// print argmin
    //cout << "argmin      " << V.transpose() << std::endl;
    //cout << endl << "f in argmin " << f(V) << std::endl;
    //ofstream outfile;
    //Eigen::Matrix<double, 1,3> out_vector;
    //outfile.open("bfgs_output.txt",std::ofstream::out);
    //for (int i=0; i<V.size()/dim; ++i)
    //{
        //To3D(V.segment<2>(2*i), out_vector);
        //outfile << setprecision(7) << out_vector(0) << '\t' <<  out_vector(1) << '\t' << out_vector(2) << '\t'<< endl;
    //}

    return 0;
}
