#include <fstream>
#include <Eigen/Dense>
#include <iostream>
#include "meta.h"
#include "problem.h"
#include "solver/bfgssolver.h"

// to use this library just use the namespace "cppoptlib"
namespace cppoptlib {

    // we define a new problem 
    // we use a templated-class rather than "auto"-lambda function for a clean architecture
        class Rosenbrock : public Problem<double> {
            public:
                // this is just the objective (NOT optional)
                double value(const Vector<double> &x) {
                    const double t1 = (1 - x[0]);
                    const double t2 = (x[1] - x[0] * x[0]);
                    return   t1 * t1 + 100 * t2 * t2;
                }

                // if you calculated the derivative by hand
                // you can implement it here (OPTIONAL)
                // otherwise it will fall back to (bad) numerical finite differences
                void gradient(const Vector<double> &x, Vector<double> &grad) {
                    grad[0]  = -2 * (1 - x[0]) + 200 * (x[1] - x[0] * x[0]) * (-2 * x[0]);
                    grad[1]  =                   200 * (x[1] - x[0] * x[0]);
                }
        };

}


using namespace std;
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

void AngleGradient(const Eigen::MatrixXd & all_angles, const int & pt_index, const double & s_power, Eigen::MatrixXd & output)
{
    Eigen::Matrix<double, 1, 3> temp_sum, temp_pt, temp_i, temp;
    Eigen::Matrix<double, 3, 2> temp_jacobian;
    temp_sum.setZero();
    for (int i=0; i<pt_index; ++i)
    {
        To3D(all_angles.row(pt_index), temp_pt);
        To3D(all_angles.row(i), temp_i);
        temp = temp_pt - temp_i;
        temp_sum += pow(temp.dot(temp), -1-s_power/2.0) * temp;
    }
    for (int i=pt_index+1; i<all_angles.rows(); ++i)
    {
        To3D(all_angles.row(pt_index), temp_pt);
        To3D(all_angles.row(i), temp_i);
        temp = temp_pt - temp_i;
        temp_sum += pow(temp.dot(temp), -1-s_power/2.0) * temp;
    }
    temp_sum *= -s_power;
    ComputeJacobian(all_angles(pt_index,0), all_angles(pt_index,1), temp_jacobian);
    output.row(pt_index) = temp_sum * temp_jacobian;
}



void FullGradient(const Eigen::MatrixXd & all_angles, const double & s_power, Eigen::MatrixXd & output)
{
    for(int i=0; i<all_angles.rows(); ++i)
    {
        AngleGradient(all_angles, i, s_power, output);
    }
}

double Energy(const cppoptlib::Vector<double> & V, const double s_power)
{
    // V contains spherical coordinates
    double e = 0;
    for (int i=0; i<V.rows(); ++i)
    {
        for (int j=0; j<i; ++j)
        {
            e += pow(dist_squared(V.segment<2>(i), V.row(j)), -s_power/2.0);
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

/////////////////////////////////////////////////////////////////////////////////
int main(int argc, char const *argv[]) {

    double s;
    int dim = 0, numpts=0;
    ifstream inputfile, pointfile;
    openFile(inputfile, "control.inp");
    ParseControlFile(inputfile, dim, numpts, s);
    inputfile.close();
    openFile(pointfile, "input.txt");
    Eigen::MatrixXd X(numpts, dim), A(numpts, dim-1), G(numpts, dim-1);
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
    Eigen::Matrix<double, 1,3> v;
    cppoptlib::Vector<double> V;
    //Eigen::MatrixXd M;
    //Eigen::Matrix<double, 3, 2> jac;

    V.transpose();
    //for (int i=0; i<3; ++i)
    //{
        //To3D(A.row(i), v);
        //cout<< X.row(i) << "      " << v << endl;
    //}
    To3D(A.row(0), v);
    V = v.transpose();
    cout << X.rows() << endl;
    cout << endl;
    V = X.col(1);
    cout << V.block(0,0, 5, 1) << endl;
    V.conservativeResize(2000);
    cout << V.size() << endl;


    //FullGradient(A, 3.0, G);
    //ComputeJacobian(A(j,0), A(j,1), jac);
    //cout << "test Energy" << endl;
    //cout << Energy(A, 3.0) << endl;
    //cout << "test FullGradient" << endl;
    //cout << G << endl;
    //cout<< G  << endl;



    // initialize the Rosenbrock-problem
    //cppoptlib::Rosenbrock<double> f;
    //// choose a starting point
    //cppoptlib::Vector<double> x(2); x << -1, 2;

    //// first check the given derivative 
    //// there is output, if they are NOT similar to finite differences
    //bool probably_correct = f.checkGradient(x);

    //// choose a solver
    //cppoptlib::BfgsSolver<double> solver;
    //// and minimize the function
    //solver.minimize(f, x);
    //// print argmin
    //std::cout << "argmin      " << x.transpose() << std::endl;
    //std::cout << "f in argmin " << f(x) << std::endl;
    return 0;
}
