# This code is provided by prof. Krzysztof Fidkowski during his class, MECHENG 523.

import numpy as np

def residual(xv):
    # this function builds the residual vector and Jacobian matrix
    x = xv[0]
    y = xv[1]
    R = np.zeros(2)  # residual vector
    R_x = np.zeros([2,2]) # residual Jacobian vector

    # residual vector: 2 x 1
    R[0] = x*x + np.sin(y) - 3   # first entry in the residual vector
    R[1] = y**3 - 2*x/y + 10     # second entry in the residual vector

    # Jacobian matrix: 2 x 2 
    R_x[0,0] =  2*x; R_x[0,1] = np.cos(y)
    R_x[1,0] = -2/y; R_x[1,1] = 3*y*y + 2*x/(y*y)

    return R, R_x


def run_newton(x0):
    print('\nInitial guess = %.3f %.3f\n'%(x0[0],x0[1]))
    x = x0.copy()  # x gets set to the initial guess
    for k in range(10):  # begin Newton iterations
        R, R_x = residual(x)  # evaluate the residual and Jacobian at x
        rnorm = np.linalg.norm(R,2)  # how big is the residual?
        print(' iteration %d: norm(R) = %.5e'%(k,rnorm))
        if (rnorm < 1e-10): break  # small enough, quit

        # in Matlab, this would be: dx = -R_x\R;
        dx = np.linalg.solve(R_x,-R) # dx = -inv(R_x)*R
        x = x + dx # take the update, and keep going for another iteration

    print('\nSolution = %.3f %.3f\n'%(x[0],x[1])) 

    return x

def main():
    x = run_newton([1.5,0.5]) # run with initial guess
    
if __name__ == "__main__":
    main()
