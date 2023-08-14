import numpy as np
from PCA import PCA
def dist_matrix(x):
    sum_x=np.sum(np.square(x),1)
    D= np.add(np.add(-2*np.dot(x,x.T),sum_x).T,sum_x)
    return D

def cal_perp(D,idx=0,beta=1.0):
    prob=np.exp(-D*beta)
    
    prob[idx]=0
    sum_prob=np.sum(prob)+1e-5
    perp = np.log(sum_prob) + beta * np.sum(D * prob) / sum_prob
    prob /= sum_prob
    return perp, prob

def prob_i(x,tol=1e-5,perplexity=30.0):
    (n,d)=x.shape
    D=dist_matrix(x)
    res_prob=np.zeros((n,n))
    beta=np.ones((n,1))
    start_perp=np.log(perplexity)

    for i in range(n):
        betamin=-np.inf
        betamax=np.inf
        perp,this_prob=cal_perp(D[i],i,beta[i])

        perp_diff=perp-start_perp
        times=0
        while np.abs(perp_diff)>tol and times<50:
            if perp_diff>0:
                betamin=beta[i].copy()
                if betamax==np.inf or betamax==-np.inf:
                    beta[i]=beta[i]*2
                else:
                    beta[i]=(beta[i]+betamax)/2
            else:
                betamax=beta[i].copy()
                if betamin==np.inf or betamin==-np.inf:
                    beta[i]=beta[i]/2
                else:
                    beta[i]=(beta[i]+betamin)/2
            perp,this_prob=cal_perp(D[i],i,beta[i])
            perp_diff=perp-start_perp
            times+=1
        res_prob[i,]=this_prob
    return res_prob

def TSNE(x,no_dims=2,initial_dims=50,perplexity=30.0, epoch=500):
    if initial_dims>50:
        x=PCA(x, 50).real
    (n,d)=x.shape
    initial_momentum=0.5
    final_momentum=0.8
    eta=300
    min_gain=0.01
    y = np.random.randn(n, no_dims)
    dy = np.zeros((n, no_dims))
    iy = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # 对称化
    P = prob_i(x, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4
    P = np.maximum(P, 1e-12)

    for i in range(epoch):
        sum_y=np.sum(np.square(y),1)
        num = 1 / (1 + np.add(np.add(-2 * np.dot(y, y.T), sum_y).T, sum_y))
        num[range(n), range(n)] = 0
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        grad=P-Q
        for j in range(n):
            dy[j,:]=np.sum(np.tile(grad[:,j]*num[:,j],(no_dims,1)).T*(y[j,:]-y),0)
        
        if i<20:
            momentum=initial_momentum
        else:
            momentum=final_momentum
        gains = (gains + 0.2) * ((dy > 0) != (iy > 0)) + (gains * 0.8) * ((dy > 0) == (iy > 0))
        gains[gains < min_gain] = min_gain
        iy = momentum * iy - eta * (gains * dy)
        y = y + iy
        y = y - np.tile(np.mean(y, 0), (n, 1))

        if (i+1)%100==0:
            if i>100:
                C=np.sum(P*np.log(P/Q))
            else:
                C = np.sum( P/4 * np.log( P/4 / Q))
            
        if i==100:
            P=P/4
    
    return y