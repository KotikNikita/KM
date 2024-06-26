import numpy as np
import scipy as sp
from scipy import optimize
import cvxopt
import src.util.utils as ut


class KernelSVC :

    def __init__(self, C, kernel, epsilon=1e-3, tol = 1e-2) :
        self.type = 'non-linear'
        self.C = C
        self.kernel = kernel
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        self.norm_f = None
        self.beyond_margin = None
        self.tol = tol
        self.mean_score = 0
        self.std_score = 1

    def fit_cvxopt(self, X, y, verbose = False, class_weights = None, precomputed_K = None) :

        N = len(y)
        d = np.diag(y)
        if precomputed_K is None:
            K = self.kernel(X, X)
        else :
            K = precomputed_K
        print('Kernel computed')
        # Constraints variables for the inequality constraint
        G = - np.kron(np.array([[-1], [1]]), np.eye(N))
        h = np.kron(np.array([self.C, 0]), np.ones(N))
        if class_weights is not None :
            h[:N] *= (y == 1)*class_weights[0] + (y == -1)*class_weights[1]
        #print("s[:N] contains ", np.unique(h[:N]))



        P = d @ K @ d
        q = -np.ones(len(K))

        A = y.reshape(1,-1).astype('float')

        b = np.array([[0]]).astype('float')

        res = ut.cvxopt_solve_qp(P,q,G,h,A,b)
        print('End optimisation ', res)

        self.alpha = d @ res
        supportIndices = np.argwhere(
            (np.abs(self.alpha) > self.epsilon) * (np.abs(self.alpha) < self.C - self.epsilon)).flatten()
        marginIndices = np.argwhere((np.abs(self.alpha) > self.epsilon)).flatten()

        self.support = X[
            supportIndices] 
        self.b = np.mean(y[supportIndices] - (K @ self.alpha)[
            supportIndices])  
        self.norm_f = self.alpha.T @ K @ self.alpha

        f = K@self.alpha + self.b
        self.mean_score = np.mean(f)
        self.std_score = np.std(f)

        self.beyond_margin = X[marginIndices]
        self.alpha = self.alpha[marginIndices]



    def fit(self, X, y, verbose = False, class_weights = None, precomputed_K = None) :
        #### You might define here any variable needed for the rest of the code
        N = len(y)
        d = np.diag(y)
        if precomputed_K is None:
            K = self.kernel(X, X)
        else :
            K = precomputed_K
        print('Kernel computed')
        # Constraints variables for the inequality constraint
        A = np.kron(np.array([[-1], [1]]), np.eye(N))
        s = np.kron(np.array([self.C, 0]), np.ones(N))
        if class_weights is not None :
            s[:N] *= (y == 1)*class_weights[0] + (y == -1)*class_weights[1]
        #print("s[:N] contains ", np.unique(s[:N]))

        # Lagrange dual problem
        def loss(alpha) :
            loss_ = 1 / 2 * alpha.T @ d @ K @ d @ alpha - np.sum(alpha)
            return loss_

        # Partial derivate of Ld on alpha
        def grad_loss(alpha) :
            grad = d @ K @ d @ alpha - np.ones_like(alpha)
            return grad

        # Constraints on alpha of the shape :
        # -  d - C*alpha  = 0
        # -  s - A*alpha >= 0

        fun_eq = lambda \
            alpha : y.T @ alpha  # '''----------------function defining the equality constraint------------------'''
        jac_eq = lambda \
            alpha : y  # '''----------------jacobian wrt alpha of the  equality constraint------------------'''
        fun_ineq = lambda \
            alpha : s + A @ alpha  # '''---------------function defining the ineequality constraint-------------------'''
        jac_ineq = lambda \
            alpha : A  # '''---------------jacobian wrt alpha of the  inequality constraint-------------------'''

        constraints = ({'type' : 'eq', 'fun' : fun_eq, 'jac' : jac_eq},
                       {'type' : 'ineq',
                        'fun' : fun_ineq,
                        'jac' : jac_ineq})
        print('Optimisation starting')
        optRes = optimize.minimize(fun=lambda alpha : loss(alpha),
                                   x0=np.ones(N),
                                   method='SLSQP',
                                   jac=lambda alpha : grad_loss(alpha),
                                   constraints=constraints,
                                   tol=self.tol,
                                   options={'disp' : verbose, 'iprint' : 2})
        print('End optimisation ', optRes)

        self.alpha = d @ optRes.x
        ## Assign the required attributes
        supportIndices = np.argwhere(
            (np.abs(self.alpha) > self.epsilon) * (np.abs(self.alpha) < self.C - self.epsilon)).flatten()
        marginIndices = np.argwhere((np.abs(self.alpha) > self.epsilon)).flatten()

        self.support = X[
            supportIndices]  # '''------------------- A matrix with each row corresponding to a support vector ------------------'''
        self.b = np.mean(y[supportIndices] - (K @ self.alpha)[
            supportIndices])  # ''' -----------------offset of the classifier------------------ '''
        self.norm_f = self.alpha.T @ K @ self.alpha  # '''------------------------RKHS norm of the function f ------------------------------'''

        # Only keep the indices where alpha is not zero
        self.beyond_margin = X[marginIndices]
        self.alpha = self.alpha[marginIndices]



    ### Implementation of the separting function $f$
    def separating_function(self, x) :
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        return self.kernel(x, self.beyond_margin) @ self.alpha + self.b

    def predict(self, X) :
        d = self.separating_function(X)
        return 2 * (d > 0) - 1


class MulticlassSVC :

    def __init__(self, nb_classes, kernel, C=1, epsilon = 1e-3, tol = 1e-2):

        self.nb_classes = nb_classes
        self.classifiers = []
        self.kernel = kernel
        self.C = C if type(C) == list else [C]*nb_classes
        for i in range(nb_classes):
            self.classifiers.append(KernelSVC(self.C[i],kernel,epsilon, tol=tol))



    def fit(self,Xtrain,Y_train, verbose= False, use_weights = True, solver = 'scipy'):

        '''Fit all the classifiers individually'''

        # Precomputed the kernel matrix
        K = self.kernel(Xtrain, Xtrain)
        print(np.linalg.matrix_rank(K))

        for i in range(self.nb_classes):
            if verbose :
                print(f'Fitting classifier {i}')
            Ybinary = 2*(Y_train == i) - 1
            svc = self.classifiers[i]
            if use_weights:
                class_weights = [len(Ybinary) / np.sum(Ybinary == 1), len(Ybinary) / np.sum(Ybinary == -1)]
            else :
                class_weights = None
            if solver == 'scipy':
                svc.fit(Xtrain, Ybinary, verbose=verbose, class_weights=class_weights, precomputed_K=K)
            else:
                svc.fit_cvxopt(Xtrain, Ybinary, verbose=verbose, class_weights=class_weights, precomputed_K=K)
        return

    def predict(self, Xtrain):


        scores = np.zeros((len(Xtrain), self.nb_classes))
        for i in range(self.nb_classes) :
            scores[:,i] = self.classifiers[i].separating_function(Xtrain)

            # Normalize so that the distribution of the scores is similar for each classifier
            scores[:,i] = (scores[:,i] - self.classifiers[i].mean_score)/(self.classifiers[i].std_score)

        classes = np.argmax(scores, axis = 1)
        return classes, scores
