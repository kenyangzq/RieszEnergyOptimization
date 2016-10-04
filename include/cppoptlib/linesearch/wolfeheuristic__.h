// CppNumericalSolver
#ifndef WOLFERULE_H_
#define WOLFERULE_H_

#include "../meta.h"

namespace cppoptlib {

    /**
     * @brief this tries to guess the correct stepwith in a bisection-style
     * @details WARNING: THIS IS QUITE HACKY. TEST ARMIJO before.
     *
     * @tparam T scalar type
     * @tparam P problem type
     * @tparam Ord order of solver
     */
    template<typename T, typename P, int Ord>
        class WolfeHeuristic {

            public:

                static T linesearch(const Vector<T> & x0, const Vector<T> & searchDir, P &objFunc, const T alpha_init = 1.0) {
                    const int IT=10;

                    // evaluate phi(0)
                    T phi0 = objFunc.value(x0);
                    // evaluate phi'(0)
                    Vector<T> grad0(x0.rows());
                    objFunc.gradient(x0, grad0);

                    T phi0_dash = searchDir.dot(grad0);
                    T alpha = alpha_init * x0.rows();
                    //unsigned int min_index=0; 
                    T min_alpha=alpha;
                    Vector<T> x_candidate = x0 + alpha * searchDir;
                    T phi = objFunc.value(x_candidate);
                    T min_phi=phi;
                    std::cout << "phi0dash " << phi0_dash << std::endl;
                    std::cout << "before linesearch energy was " << phi0 << std::endl;
                    for (unsigned int iter=0; iter<IT; ++iter)
                    {
                        if (phi > phi0 + 0.0002 * alpha * phi0_dash)
                            alpha *= 0.8;
                        else 
                        {
                            alpha *= 2.0;
                            //std::cout << "increasing alpha" << std::endl;
                        }
                        x_candidate = x0 + alpha * searchDir;
                        phi = objFunc.value(x_candidate);
                        if (phi < min_phi)
                        {
                            //min_index = iter;
                            min_alpha = alpha;
                            min_phi = phi;
                        }
                    }

                    //while (phi > phi0 + 0.0001 * alpha * phi0_dash)
                    //{
                        //alpha *= 0.4;
                        //x_candidate = x0 + alpha * searchDir;
                        //phi = objFunc.value(x_candidate);
                    //}
                    //T alpha_backup = alpha;
                    //Vector<T> grad_candidate(x0.rows());
                    //objFunc.gradient(x_candidate,grad_candidate);
                    //T phi_dash = searchDir.dot(grad_candidate);
                    //for (unsigned int iter=0; iter<5 && (phi_dash < 0.9*phi0_dash); ++iter)
                    //{
                            //alpha_backup = alpha;
                            //alpha *= 2.0;
                            //x_candidate = x0 + alpha * searchDir;
                            //objFunc.gradient(x_candidate,grad_candidate);
                            //phi_dash = searchDir.dot(grad_candidate);
                    //}
                    std::cout << "after it became  " << min_phi << std::endl;
                    return min_alpha;
                }

        };
}

#endif /* WOLFERULE_H_ */
