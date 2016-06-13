// TODO add multiprecision arithmetic with either https://gmplib.org/, or http://www.mpfr.org/
// TODO use an autodifferentiation library like CppAD, http://www.coin-or.org/CppAD/
//
//
//
#include <iomanip>
#include <fstream>
#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include "meta.h"
#include "problem.h"
#include "solver/lbfgsbsolver.h"
#include "solver/lbfgssolver.h"
#include <omp.h>
#define PI 3.141592653589793
#define THREADNUM 2

using namespace std;

    // we define a new problem 
    // we use a templated-class rather than "auto"-lambda function for a clean architecture

void openFile (ifstream & inputfile, string name){
    inputfile.open(name.c_str());
    if (inputfile.fail()) {
        cout << "Error opening input data file\n";
        exit(1);
    }
}

string ParseControlFile(ifstream & inputfile, int &dim, int &numpts, double & s, int & c, int & max_neighbor, int & history_size){
    
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
        }else if (lineNumber == 6) {
            tmp >> k >> k >> c;
        }else if (lineNumber == 7) {
            tmp >> k >> k >> infile;
        }else if (lineNumber == 8 && !infile) {
            tmp >> k >> k >> numpts;
        }else if (lineNumber == 10) {
            tmp >> k >> k >> numpfiles;
        }else if (lineNumber == 11) {
            tmp >> k >> k >> max_neighbor;
        }else if (lineNumber == 12) {
            tmp >> k >> k >> filename;
        }
    }
    cout << "Summary of the control file:\n\n";
    cout << "S value: " << s << "\n";
    cout << "Dimension: " << dim << "\n";
    cout << "C value: " << c << "\n";
    cout << "Infile request: " << infile << "\n";
    if (!infile) cout << "Number of points: " << numpts << "\n";
    cout << "Number of output files: " << numpfiles << "\n";
    cout << "Max neighbor: " << max_neighbor << "\n";
    cout << "LBFGS history size: " << history_size  << "\n";
    cout << "Output filename: " << filename << ".txt\n\n\n\n";
    return filename;
}

double dist_squared(const cppoptlib::Vector<double> & angles1,const cppoptlib::Vector<double> & angles2)
{
    return (2-2*(sin(angles1(1))*sin(angles2(1))*cos(angles1(0)-angles2(0))+cos(angles1(1))*cos(angles2(1))));
}

void To3D(const cppoptlib::Vector<double> & angles, cppoptlib::Vector<double> & coords)
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

void ComputeJacobian(const double & theta, const double & phi, cppoptlib::Matrix<double> & temp){
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
    cppoptlib::Vector<double> temp_sum(3), temp_pt(3), temp_i(3), temp(3);
    cppoptlib::Matrix<double> temp_jacobian(3,2);
    temp_sum.setZero();
    To3D(all_angles.segment<2>(pt_index*2), temp_pt);
    for (int i=0; i<pt_index; ++i)
    {
        To3D(all_angles.segment<2>(i*2), temp_i);
        temp = temp_pt - temp_i;
        temp_sum += pow(temp.dot(temp), -1-s_power/2.0) * temp;
    }
    for (int i=pt_index+1; i<all_angles.rows()/2; ++i)
    {
        To3D(all_angles.segment<2>(i*2), temp_i);
        temp = temp_pt - temp_i;
        temp_sum += pow(temp.dot(temp), -1-s_power/2.0) * temp;
    }
    temp_sum *= -s_power;
    //cout << "here is your sum: "<< temp_sum << endl;
    ComputeJacobian(all_angles(pt_index*2+0), all_angles(pt_index*2+1), temp_jacobian);
    output.segment<2>(pt_index*2) = temp_sum.transpose() * temp_jacobian;
}

void FullGradient(const cppoptlib::Vector<double> & all_angles, const double & s_power, cppoptlib::Vector<double> & output)
{
    for(int i=0; i<all_angles.size()/2; ++i)
    {
        AngleGradient(all_angles, i, s_power, output);
    }
}

double EnergyMatrix(const cppoptlib::Matrix<double> & M, const double s_power)
    //computes energy using matrices
{
    // M contains spherical coordinates
    double e = 0;
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
    double e = 0.0;
    double e_threaded[THREADNUM]={0.0};
    int i, j;

 #pragma omp parallel shared(V, e_threaded) private(i, j)
    {
#pragma omp for schedule(dynamic)
        for (i=0; i<V.size()/dim; ++i)
        {
            for (j=0; j<i; ++j)
            {
                e_threaded[omp_get_thread_num()] += pow(dist_squared(V.segment(i*dim, dim), V.segment(j*dim, dim)), -s_power/2.0);
            }
        }
    }
    // end of parallel code
    for (i=0; i<THREADNUM; ++i)
        e += e_threaded[i];
    return 2.0 * e;
}

double PointEnergy(const cppoptlib::Vector<double> & V, const int pt_index, const double s_power, const int dim)
{
    double e = 0;
    for (int i=0; i<pt_index; ++i)
    {
        e += pow(dist_squared(V.segment(i*dim, dim), V.segment(pt_index*dim, dim)), -s_power/2.0);
    }
    for (int i=pt_index+1; i<V.size()/dim; ++i)
    {
        e += pow(dist_squared(V.segment(i*dim, dim), V.segment(pt_index*dim, dim)), -s_power/2.0);
    }
    return e;
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



void writeFile (ofstream & outputfile, string name, cppoptlib::Vector<double> V, int dim){
    outputfile.open(name.c_str());
    if (outputfile.fail()) {
        cout << "Error writing output data file" << endl;
        exit(1);
    }
    int c = dim-1;
    
    outputfile << setprecision(6);
    outputfile << fixed;
    
    for (int i =0; i < V.rows()/c; i++) {
        cppoptlib::Vector<double> tmp = V.segment(i*c, c);
        cppoptlib::Vector<double> tmp2;
        To3D(tmp, tmp2);
        outputfile << tmp2(0) << "\t" << tmp2(1) << "\t" << tmp2(2) << "\n";
    }
}


        class DemoProblem : public cppoptlib::Problem<double> {
            int dim;
            double s;
            public:
            DemoProblem(int dim_value, double s_value):dim(dim_value),s(s_value){}
                double value(const cppoptlib::Vector<double> &x) {
                    return Energy(x, s, dim-1);
                }
                void gradient(const cppoptlib::Vector<double> &x, cppoptlib::Vector<double> &grad) {
                    FullGradient(x, s, grad);
                }
        };
/////////////////////////////////////////////////////////////////////////////////
int main(int argc, char const *argv[]) {
    
    
    omp_set_num_threads(THREADNUM);   
    double s, radius;
    int dim = 0, numpts=0, c=0, cubes_per_side=0, max_neighbor=0, history_size=5;
    ifstream inputfile, pointfile;
    ofstream outputfile;

    openFile(inputfile, "control.inp");
    string filename = ParseControlFile(inputfile, dim, numpts, s, c, max_neighbor, history_size);
    
    radius = c*pow(numpts,-1.0/(dim-1));
    cubes_per_side = ceil(2/radius);
    
    inputfile.close();
    openFile(pointfile, filename+".txt");
    Eigen::MatrixXd X(numpts, dim), A(numpts, dim-1);
    cppoptlib::Vector<double> V(A.size()), G(A.size()), GFull(A.size()), energies(A.rows());
    
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
    ToVector(A,V);
    //Eigen::Matrix<double, 1,3> v;

     //initialize the DemoProblem-problem
     //
     //
    cout << "  original energy: " << Energy(V, s, 2) << endl;
    DemoProblem f(dim, s);
    int Dim = dim-1;

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

    cppoptlib::LbfgsbSolver<double> solver;
    //cppoptlib::LbfgsSolver<double> solver;
    solver.minimize(f, V);
    //// print argmin
    cout << "argmin      " << V.transpose() << std::endl;
    cout << endl << "f in argmin " << f(V) << std::endl;
    ofstream outfile;
    cppoptlib::Vector<double> out_vector(3);
    outfile.open("bfgs_output.txt",std::ofstream::out);
    std::ptrdiff_t l, k;
    for (int i=0; i<V.size()/Dim; ++i)
    {
        To3D(V.segment<2>(2*i), out_vector);
        outfile << setprecision(7) << out_vector(0) << '\t' <<  out_vector(1) << '\t' << out_vector(2) << '\t'<< endl;
        energies(i) = PointEnergy(V, i, s, dim-1);
        outfile << setprecision(7) << energies(i) << endl;
    }
    cout << endl << "Here is mean():      " << energies.mean()      << endl;
    cout << "Here is minCoeff():  " << energies.minCoeff(& l)  << " at " << l << endl;
    cout << "Here is maxCoeff():  " << energies.maxCoeff(& k)  << " at " << k << endl;
//    cout << energies.transpose() << endl;
//writeFile(outputfile, filename+"bfgs.txt", V, dim);

    return 0;
}
