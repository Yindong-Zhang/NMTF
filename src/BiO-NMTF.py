import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
import argparse
import time
import csv
def check_stop(hist, epsilon= -7, wait= 2):
    EPS= 10**epsilon
    flag= 1
    for i in range(-wait, 0):
        prev= hist[i-1]
        now= hist[i]

        if prev - now > now * EPS :
            flag= 0
            break
    return flag

def dump_log(configStr, err_history):
    filename= '../logs/%s.txt' %(configStr, )
    with open(filename, 'w') as fp:
        writer = csv.writer(fp, delimiter=' ')
        for i, h in enumerate(err_history):
            writer.writerow([i, h])



def multiplicative_update(X, F, S, G, min_iter= 100, max_iter= 1000, epsilon= -6, tolerance= 5, verbose= True):
    err_history= []
    trXX= np.trace(np.dot(X.T, X))

    for i in range(max_iter):

        # update G
        numerator= reduce(np.dot, [X.T, F, S])
        denominator= reduce(np.dot, [G, G.T, numerator])
        mu_G= np.sqrt(numerator/denominator)
        G= G * mu_G

        # update F
        numerator= reduce(np.dot, [X, G, S.T])
        denominator= reduce(np.dot, [F, F.T, numerator])
        mu_F= np.sqrt(numerator/denominator)
        F= F * mu_F

        # update S
        enumrator= FtXG= reduce(np.dot, [F.T, X, G])
        FtF= np.dot(F.T, F)
        GtG= np.dot(G.T, G)
        denominator=reduce(np.dot, [FtF, S, GtG])
        mu_S= np.sqrt(enumrator/denominator)
        S= S * mu_S

        # compute loss
        tr2= np.trace(np.dot(FtXG.T, S))
        tr3= np.trace(reduce(np.dot, [FtF, S, GtG, S.T]))
        loss= trXX - 2*tr2 + tr3
        err_history.append(loss)

        if i >= min_iter and check_stop(err_history, epsilon, wait= tolerance):
            print("Stopping After %d iteration!" %(i, ))
            break

        if verbose:
            print("loss: %s" %(loss, ))

    return (F, S, G), err_history

def validate_factors(factors):
    for f in factors:
        if np.any(f < 0):
            raise Exception("Assert exception: factor contains negative values")



def tri_factorization(X, F, S, G, min_iter= 5, max_iter= 1000, epsilon= -6, tolerance= 5, verbose= True):
    t0= time.time()
    factors, losses= multiplicative_update(X, F, S, G, min_iter, max_iter, epsilon, tolerance, verbose)
    validate_factors(factors)
    t1= time.time()
    duration= t1 - t0
    if verbose:
        print('Losses histroy: %s' %(losses, ))
        print('Time %s cost.' %(duration, ))
    return factors, losses


def load_data():
    pass

def load_test():
    return 1000 * np.random.rand(500, 1000)

def main():
    parser= argparse.ArgumentParser(description= 'non-negative matrix tri-factorization')
    parser.add_argument('-row', '--rank_row', type= int, default= 3,
                        help= "factorization rank(row dimension)")
    parser.add_argument('-col', '--rank_column', type= int, default= 3,
                        help= "factorization rank(column dimension)")
    parser.add_argument('-i', '--max_iter', type=int, default=20, help="Maximum number of iterations.")
    parser.add_argument('-m', '--min_iter', type=int, default=6, help="Specify minimum number of iterations.")
    parser.add_argument('-e', '--epsilon', type= int, default= -10,
                        help= "Convergence criteria: noise flunctuation at convergence should be less 10**epsilon.")
    parser.add_argument('-t', '--tolerance', type= int, default= 5,
                        help= "Loss decrease tolerance when judging convergence.")
    parser.add_argument('-r', '--repeat', type= int, default= 6,
                        help= 'Repeat count in regard of initilization dependency.')
    parser.add_argument('-l', '--label', type= str, default= 'test',
                        help= "label of this run case for discriminitive output filename.")
    args= parser.parse_args()
    configStr= 'label~%s-rank_row~%s-rank_col~%s-epsilon~%s-tolerance~%s' %(args.label, args.rank_row, args.rank_column, args.epsilon, args.tolerance)

    data= load_test()
    # data= load_data()
    # low rank shape
    # m * n = m * d ^ d * e ^ e * n
    assert len(data.shape)== 2, "Input data is not a 2 dimensional matrix"
    assert args.tolerance < args.min_iter, "Convergence tolerance should be smaller than min_iter"
    m= data.shape[0]
    n= data.shape[1]
    d= args.rank_row
    e= args.rank_column
    print('Data shape: %s' %(data.shape, ))
    print('Low rank Matrix Shape: (%s, %s)' %(d, e))
    factors_list= []
    losses_list= []
    for it in range(args.repeat):
        print("Start the %s-th try..." %(it, ))
        F= 10 * np.random.rand(m, d)
        S= 10 * np.random.rand(d, e)
        G= 10 * np.random.rand(n, e)
        params= {
            'min_iter': args.min_iter,
            'max_iter': args.max_iter,
            'epsilon': args.epsilon,
            'tolerance': args.tolerance,
            'verbose': True
        }

        factors, losses= tri_factorization(data, F, S, G, **params)

        configDetail= configStr + '-%s' %(it, )
        dump_log(configDetail, losses)

        factors_list.append(factors)
        losses_list.append(losses)

    min_losses= np.min(losses_list, -1)
    ind= np.argmin(min_losses)
    losses= losses_list[ind]
    factors=factors_list[ind]
    return factors, losses

if __name__ == '__main__':
    factors, losses= main()
    plt.plot(losses)
    plt.show()