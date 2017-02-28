#include "min_energy.h"
//TODO: timing
//TODO: torus
//TODO: I/O
//
//
using namespace std;

int main(int argc, char const *argv[]) {
    
    double s, radius;
    int dim = 0, numpts=0, c=0, cubes_per_side=0, max_neighbor=0, history_size=5;
    bool infile;
    ifstream inputfile, pointfile, comparefile;
    ofstream outputfile;
    int numcubes = pow(3, dim);

    
    openFile(inputfile, "control.inp");
    string filename = ParseControlFile(inputfile, dim, numpts, s, c, max_neighbor, history_size);
    inputfile.close();
    
    radius = c*pow(numpts,-1.0/(dim-1));
    cubes_per_side = ceil(2/radius);
    
    cout << "filename is: " << filename << endl;
    //openFile(comparefile, "jun7test500-final.txt");
    Eigen::MatrixXd  X(numpts, dim), A(numpts, dim-1);
    cppoptlib::Vector<double> V(A.size()), G(A.size()), GFull(A.size());
    
    // read points
    if (infile)
    {
        openFile(pointfile, filename+".txt");
        int lineNumber = 0;
        while (!pointfile.eof() && lineNumber < numpts)
        {
            for (int i=0; i<dim;  ++i)
            {
                pointfile >> X(lineNumber, i);
            }
            lineNumber++;
        }
        pointfile.close();
    }
    else
    {
        srand(time(0));
        double apoint[dim];
        for (int i = 0; i < numpts; i++) {
            randptSphere(apoint, dim);
            for (int j = 0; j < dim; j++) {
                X(i, j) = apoint[j];
            }
        }
    }
    ToAngles(X,A, dim);
    Flatten(A,V);
    minimizeEnergy f(radius, s, dim, numpts, cubes_per_side, pow(cubes_per_side, dim), max_neighbor);
    cout << "Energy without cutoff:      " << Energy(V, s, dim-1) << endl;
    cout << "Energy with cutoff:         " << f(V) << endl;
    //cout << V << endl;
    //f.gradient(V,G);
    //cout << G << endl;
    //FullGradient(V, s, GFull);

    //cout << "gradient with cutoff  "<< endl << G.transpose()<< endl << endl << endl;
    //cout << "difference of gradients" << GFull.transpose() - G.transpose() << endl;

    
    
    
    cppoptlib::LbfgsSolver<minimizeEnergy> solver;
    //solver.setHistorySize(history_size);
    solver.minimize(f, V);

    //ToAngles(ComparePoints,A);
    //Flatten(A,V);
    //cout << "Energy produced by the C-code: " << Energy(V, s, dim-1) << endl;
    
    cout << "Cutoff energy after:   " << f(V) << endl;
    cout << "Full energy after:     " << Energy(V, s, dim-1) << endl;
    //writeFile(outputfile, filename+"bfgs.txt", V, dim);
    return 0;
}
