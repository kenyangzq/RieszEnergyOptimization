#include <fstream>
#include <Eigen/Dense>
#include <iostream>
#include "meta.h"
#include "problem.h"
#include "solver/bfgssolver.h"

// to use this library just use the namespace "cppoptlib"
namespace cppoptlib {

    // we define a new problem for optimizing the rosenbrock function
    // we use a templated-class rather than "auto"-lambda function for a clean architecture
    template<typename T>
        class Rosenbrock : public Problem<T> {
            public:
                // this is just the objective (NOT optional)
                T value(const Vector<T> &x) {
                    const T t1 = (1 - x[0]);
                    const T t2 = (x[1] - x[0] * x[0]);
                    return   t1 * t1 + 100 * t2 * t2;
                }

                // if you calculated the derivative by hand
                // you can implement it here (OPTIONAL)
                // otherwise it will fall back to (bad) numerical finite differences
                void gradient(const Vector<T> &x, Vector<T> &grad) {
                    grad[0]  = -2 * (1 - x[0]) + 200 * (x[1] - x[0] * x[0]) * (-2 * x[0]);
                    grad[1]  =                   200 * (x[1] - x[0] * x[0]);
                }

                // same for hessian (OPTIONAL)
                // if you want ot use 2nd-order solvers, I encourage you to specify the hessian
                // finite differences usually (this implementation) behave bad
                void hessian(const Vector<T> &x, Matrix<T> & hessian) {
                    hessian(0, 0) = 1200 * x[0] * x[0] - 400 * x[1] + 1;
                    hessian(0, 1) = -400 * x[0];
                    hessian(1, 0) = -400 * x[0];
                    hessian(1, 1) = 200;
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

double dist(const cppoptlib::Vector<double> & angles1,const cppoptlib::Vector<double> & angles2)
{
    return sqrt(2-2*(sin(angles1(1))*sin(angles2(1))*cos(angles1(0)-angles2(0))+cos(angles1(1))*cos(angles2(1))));
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

//void ToAngles(Eigen::MatrixXd & all_points, Eigen::MatrixXd & all_angles) 

//void PointGradient(const Eigen::MatrixXd & all_points, const int & pt_index, const double & s_power, Eigen::Matrix<double, 1, 3> & output)
//{
    //Eigen::Matrix<double, 1, 3> temp_sum, temp;
    //output << 0., 0., 0.;
    //for (int i=0; i<pt_index; ++i)
    //{
        //temp = all_points.row(pt_index) - all_points.row(i);
        //output += pow(temp.dot(temp), -1-s_power/2.0) * temp;
    //}
    //for (int i=pt_index+1; i<all_points.rows(); ++i)
    //{
        //temp = all_points.row(pt_index) - all_points.row(i);
        //output += pow(temp.dot(temp), -1-s_power/2.0) * temp;
    //}
    //output *= -s_power;
//}


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
    openFile(pointfile, "short.txt");
    Eigen::MatrixXd X(numpts, dim), A(numpts, dim-1), G(numpts, dim-1);
    //
    // testing
    //
    int lineNumber = 0;
    while (!pointfile.eof() && lineNumber < numpts)
    {
        for (int i=0; i<3;  ++i)
        {
            pointfile >> X(lineNumber, i);
        }
        lineNumber++;
    }

    ToAngles(X,A);
    Eigen::Matrix<double, 1,3> v;
    cppoptlib::Vector<double> V;
    Eigen::MatrixXd M;
    Eigen::Matrix<double, 3, 2> jac;

    V.transpose();
    for (int i=0; i<3; ++i)
    {
        To3D(A.row(i), v);
        cout<< X.row(i) << "      " << v << endl;
    }
    To3D(A.row(0), v);
    V = v.transpose();
    cout << endl;
    FullGradient(A, 3.0, G);
    int j=3;
    //ComputeJacobian(A(j,0), A(j,1), jac);
    cout << "test AngleGradient" << endl;
    cout<< G  << endl;



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
    pointfile.close();
    return 0;
}
