#include "min_energy.h"
using namespace std;


void openFile (ifstream & inputfile, string name){

//
//
//
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
    cout << "\nSummary of the control file:\n\n";
    cout << "S value: " << s << "\n";
    cout << "Dimension: " << dim << "\n";
    cout << "C value: " << c << "\n";
    cout << "Infile request: " << infile << "\n";
    cout << "Number of points: " << numpts << "\n";
    cout << "Max neighbor: " << max_neighbor << "\n";
    if (infile) cout << "Input filename: " << filename << "\n\n";
    else cout << "No input file request; program will generate a random configuration.\n\n";
    cout << "LBFGS history size: " << history_size  << "\n";
    cout << "Output filename: " << filename << ".txt\n\n\n\n";
    return filename;
}


void To3D(const cppoptlib::Vector<double> & angles, cppoptlib::Vector<double> & coords)
{
    coords(0) = cos(angles(0)) * sin(angles(1));
    coords(1) = sin(angles(0)) * sin(angles(1));
    coords(2) = cos(angles(1));
}

void ToND(const cppoptlib::Vector<double> & angles, cppoptlib::Vector<double> & coords) 
{
    double tmp = 1;
    coords[0] = cos(angles[0]);
    for (int i = 1; i < coords.size()-1; i++) {
	   	tmp *= sin(angles[i-1]);
        coords[i] = cos(angles[i])*tmp;
//        cout << coords[i] << endl;
    }
    coords[coords.size()-1] = tmp*sin(angles[angles.size()-1]);
}

void Flatten(const Eigen::MatrixXd & M, cppoptlib::Vector<double> & V )
{
    int c = M.cols();
    for (int i=0; i<M.rows(); ++i )
        V.segment(i*c,c) = M.row(i);
}

double cutoff(const double & distance, const double & cut)
{
    return pow(1- pow(distance/cut,4.0),3.0);
}


double cutoff_prime(const double  & distance, const double  & cutoff)
{
    double ratio = distance/cutoff;
    return  (- 12.0 * pow(1- pow(ratio, 4.0), 2.0) * pow(ratio, 3.0))/cutoff;
}

void ComputeJacobian(const double & theta, const double & phi, cppoptlib::Matrix<double> & temp)
{
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

void ToAngle(cppoptlib::Vector<double> & angels, const cppoptlib::Vector<double> & coords) 
{
    int dim = coords.size();
    double squaresum = pow(coords[dim - 1],2);
    
    for (int i = dim - 2; i >= 0; i--) {
        squaresum += pow(coords[i], 2);
        angels[i] = acos(coords[i]/sqrt(squaresum));
    }
    
    if (coords[dim-1] < 0) {
        angels[dim-2] = 2 * PI - angels[dim-2];
    }
    //cout << angels.transpose() << endl;
}

void AngleGradient(const cppoptlib::Vector<double> & all_angles, const int & pt_index, const double & s_power, cppoptlib::Vector<double> & output)
{
    cppoptlib::Vector<double> temp_sum(3), temp_pt(3), temp_i(3), temp(3);
    cppoptlib::Matrix<double> temp_jacobian(3,2);
    temp_sum.setZero();
    ToND(all_angles.segment<2>(pt_index*2), temp_pt);
    for (int i=0; i<pt_index; ++i)
    {
        
        ToND(all_angles.segment<2>(i*2), temp_i);
        temp = temp_pt - temp_i;
        temp_sum += pow(temp.dot(temp), -1-s_power/2.0) * temp;
    }
    for (int i=pt_index+1; i<all_angles.rows()/2; ++i)
    {
        ToND(all_angles.segment<2>(i*2), temp_i);
        temp = temp_pt - temp_i;
        temp_sum += pow(temp.dot(temp), -1-s_power/2.0) * temp;
    }
    temp_sum *= -s_power;
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


void ComputeJacobianN(const cppoptlib::Vector<double> & angles, cppoptlib::Matrix<double> &temp) 
{
	// all angles range from 0 to PI except for the last one, which range from 0 to 2PI.
	// The Jacobian Matrix should be size n by n-1;
	// Multiply by a -tan value if it's differentiating cos
	// Multiply by a cot value if it's differentiating sin
	//
	// Note that the Jacobian Matrix in this case is lower diagonal
	//
	int n = angles.size() + 1;
	temp = cppoptlib::Matrix<double>::Zero(n,n-1);
	cppoptlib::Vector<double> ndvector(n);
	ToND(angles,ndvector);
	// main loop
	for (int i = 0; i < n-1; i++) 
    {
		// calculate the diagonal value
		temp(i,i) = ndvector(i)*(-tan(angles(i)));
		for(int j = 0; j < i; j++) 
        {
			// calculate the part in the lower diagonal matrix
			// except the last row of the matrix
			temp(i,j) = ndvector(i)/tan(angles(j));
		}
		// do the last row of the matrix here, separately
		temp(n-1,i) = ndvector(n-1)/tan(angles(i));
	}
}

double dist_squared(const cppoptlib::Vector<double> & angles1,const cppoptlib::Vector<double> & angles2)
{
    cppoptlib::Vector<double> temp2(angles2.size()+1), temp1(angles1.size()+1);
    ToND(angles2, temp2);
    ToND(angles1, temp1);
    return (temp2- temp1).squaredNorm();
}


//double EnergyMatrix(const cppoptlib::Matrix<double> & M, const double s_power)
//{
//    // M contains spherical coordinates
//    double e = 0;
//    for (int i=0; i<M.rows(); ++i)
//    {
//        for (int j=0; j<i; ++j)
//        {
//            e += pow(dist_squared(M.row(i), M.row(j)), -s_power/2.0);
//        }
//    }
//    return 2.0 * e;
//}

void ToAngles(cppoptlib::Matrix<double> & all_points, cppoptlib::Matrix<double> & all_angles, int dim)
{
 
	cppoptlib::Vector<double> apoint(dim);
    cppoptlib::Vector<double> aangle(dim-1);
	for (int i = 0; i < all_points.rows(); i++) {
        apoint = all_points.row(i);
        ToAngle(aangle, apoint);
        all_angles.row(i) = aangle;
    }
}

double Energy(const cppoptlib::Vector<double> & V, const double s_power, const int dim)
{
    // V contains spherical coordinates
    double e = 0;
    for (int i=0; i<V.size()/dim; ++i)
    {
        for (int j=0; j<i; ++j)
        {
            e += pow(dist_squared(V.segment(i*dim, dim), V.segment(j*dim, dim)), -s_power/2.0);
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

void writeFile (ofstream & outputfile, string name, cppoptlib::Vector<double> V, int dim)
{
    outputfile.open(name.c_str());
    if (outputfile.fail()) {
        cout << "Error writing output data file" << endl;
//        exit(1);
    }
    int c = dim-1;
    
    outputfile << setprecision(6);
    outputfile << fixed;
    
    cppoptlib::Vector<double> tmp2(dim);
    cppoptlib::Vector<double> tmp(c);
    
    for (int i =0; i < V.rows()/c; i++) {
        tmp = V.segment(i*c, c);
        ToND(tmp, tmp2);
        
        
        outputfile << tmp2(0);
        for (int j = 1; j < dim; j++) {
            outputfile << "\t" << tmp2(j);
        }
        outputfile << "\n";
    }
}

double minimizeEnergy::TruncatedEnergy(const cppoptlib::Vector<double> & V, const cppoptlib::Vector<int> & neighbors, const int index)
{
    double energy = 0;
    int c = dim -1;
    for (int i = 0; i < neighbors.size(); ++i) 
    {
        int tmp = neighbors(i); // goes over all neighbor cubes
        if (tmp != -1) // check if there is a neighbor in this direction
        { 
            int points_in_cube_number = Cubes(0, tmp);
            for (int j = 1; j <= points_in_cube_number; j++) {
                int point_index = Cubes(j, tmp);
                double distance = dist_squared(V.segment(index*c, c), V.segment(point_index*(c), c));
                if (point_index != index && distance <= cutoff_radius*cutoff_radius)
                    energy += pow(distance, -s/2.0) * cutoff(sqrt(distance), cutoff_radius) ;
            }
        }
    }
    return energy;
}

int minimizeEnergy::Point2Cube(const cppoptlib::Vector<double> & thisPt)
{
    int tmp = 0, output = 0;
    for(int i = 0; i < dim; i++){
        tmp = floor((thisPt(i) +1.0) / cutoff_radius);
        output += tmp * pow(cubes_per_side, i);
    }
    return output;
}


void minimizeEnergy::FindNeighborCubes(int cube_index, cppoptlib::Vector<int> & neighbors)
{
    
    cppoptlib::Vector<int> current_index(dim);
    //neighbors = Eigen::MatrixXd::Constant(neighbors.size(), 1, -1);
    int tmp = cube_index;
    
    for (int i = 0; i < dim; ++i){
        current_index(i) = tmp % cubes_per_side;
        tmp = tmp / cubes_per_side;
    }
    
    for (int i = 0; i < pow(3, dim); i++){
        int step = i;
        int flag = 0, index = 0;
        // index is the index of neighboring cube in the matrix of cubes;
        
        for (int j = 0; j < dim && flag == 0; j++){
            int k = step%3 - 1 + current_index[j];
            step = step/3;
            if (k < 0 || k >= cubes_per_side) flag=1;
            index += k * pow(cubes_per_side, j);
        }
        if (flag == 0) neighbors[i] = index;
    }
}

double minimizeEnergy::value(const cppoptlib::Vector<double> &x) 
    {
        double total_energy = 0;
        //int max_neighbor = Cubes.rows();
        cppoptlib::Vector<int> neighbor_cube_indices (pow(3, dim));
        // get the 3d points matrix -- pts
        // assign points to cubes
        // the first field in each column is the number of points inside that column.
        BuildIndex(x);
        
        // find the neighbor_cube_indices for each cube and calculate the energy.
        for (int i = 0; i < Cubes.cols(); ++i) 
        {
            neighbor_cube_indices = cppoptlib::Vector<int>::Ones(pow(3, dim))*(-1);
            FindNeighborCubes(i, neighbor_cube_indices);
            //cout << neighbor_cube_indices.transpose() << endl << endl;
            int points_in_cube_number = Cubes(0, i);
            // j goes over all points in the i-th cube
            for (int j = 1; j <= points_in_cube_number; ++j) {
                int point_index = Cubes(j, i); // absolute point index
                total_energy += TruncatedEnergy(x, neighbor_cube_indices, point_index);
            }
        }
        return total_energy;
    }

void minimizeEnergy::gradient(const cppoptlib::Vector<double> &x, cppoptlib::Vector<double> &grad) 
{
        int dim_angle = dim-1;
        int max_neighbor = Cubes.rows();
        double distance;
        cppoptlib::Vector<int> neighbor_cube_indices (pow(3, dim));
        cppoptlib::Vector<double> temp_sum(dim), temp(dim);
        cppoptlib::Matrix<double> temp_jacobian(dim, dim_angle);
        temp_sum.setZero();
		BuildIndex(x);
        
        // find the neighbor_cube_indices for each cubes and calculate the energy.
        for (int index_cube = 0; index_cube < Cubes.cols(); ++index_cube)  
        {
            neighbor_cube_indices = cppoptlib::Vector<int>::Ones(pow(3, dim))*(-1);
            FindNeighborCubes(index_cube, neighbor_cube_indices);

            int points_in_cube = Cubes(0, index_cube);
            for (int j = 1; j <= points_in_cube; ++j) 
            {
                temp_sum.setZero();
                int point_index = Cubes(j, index_cube);
                for (int k = 0; k < neighbor_cube_indices.size(); ++k) 
                {
                    int tmp = neighbor_cube_indices(k);
                    if (tmp != -1) 
                    {
                        int points_in_other_cube = Cubes(0, tmp);
                        for (int l = 1; l <= points_in_other_cube; l++) 
                        {
                            int other_point_index = Cubes(l, tmp);
                            temp = ptsND.col(point_index) - ptsND.col(other_point_index);
                            distance = sqrt(temp.dot(temp));
                            if (other_point_index != point_index && distance < cutoff_radius)
                            {
                                temp_sum += (cutoff_prime(distance, cutoff_radius)*pow(distance, -s) +
                                (-s)*cutoff(distance,cutoff_radius)*pow(distance, -s-2))*temp;
                            }
                        }
                    }
                }
                //
                temp_jacobian.setZero();
                ComputeJacobianN(x.segment(point_index*dim_angle,dim_angle), temp_jacobian);
                
                grad.segment(point_index*dim_angle, dim_angle) = temp_sum.transpose() * temp_jacobian;
            }
            
        }
    }


void minimizeEnergy::BuildIndex(const cppoptlib::Vector<double> & x)
{
// get the 3d points matrix -- pts
// assign points to cubes
// the first field in each column is the number of points inside that column.
    Cubes.row(0).setZero();
    
    cppoptlib::Vector<double> temp_vector(dim);
    for (int i=0; i<numpts; ++i)
    {
        ToND(x.segment((dim-1)*i, dim-1), temp_vector);
        ptsND.col(i) = temp_vector.transpose();
        int max_neighbor = Cubes.rows();
        int cube_index = Point2Cube(ptsND.col(i));
        int tmp_numpts = Cubes(0, cube_index) + 1;
        if (tmp_numpts >= max_neighbor) {
            cout << "Warning: exceeding maximum neighbor; ignore point " << i << endl;
        }else{
            Cubes(tmp_numpts, cube_index) = i;
            Cubes(0, cube_index) = tmp_numpts;
        }
    }
    //cout << Cubes.transpose() << endl;
    //cout << Point2Cube(ptsND.col(4)) << endl;
}

// generate random sphere configuration
void randptSphere(double coordinates[], int dim)
{
    
    double z;
    double norm;
    double normsq=2;
    
    while(normsq>1 || normsq==0){
        normsq=0;
        
        for(int i=0;i<dim;i++){
            z=1-(2*(double)rand()/(double)RAND_MAX);
            normsq += z*z;
            coordinates[i] = z;
        }
    }
    
    norm=sqrt(normsq);
    
    for(int i=0;i<dim;i++){
        coordinates[i] = coordinates[i]/norm;
    }
    
}



/////////////////////////////////////////////////////////////////////////////////
