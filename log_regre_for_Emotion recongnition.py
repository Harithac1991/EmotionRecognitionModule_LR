import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def getdata(balance_ones=True):
    Y = []
    X = []
    first = True
    for line in open('fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X, Y = np.array(X) / 255.0, np.array(Y)

    if balance_ones:
        # balance the 1 class
        X0, Y0 = X[Y!=1, :], Y[Y!=1]
        X1 = X[Y==1, :]
        X1 = np.repeat(X1, 9, axis=0)
        X = np.vstack([X0, X1])
        Y = np.concatenate((Y0, [1]*len(X1)))
	return(X,Y)

def y2indicator(y):
	N = len(y)
	K = len(set(y))
	ind = np.zeros((N, K))
	for i in range(N):
	    ind[i, y[i]] = 1
	return (ind)
def softmax(A):
    expA = np.exp(A)
    return(expA / expA.sum(axis=1, keepdims=True))

def get_cost(X,T):
	return(-(T *np.log(Y)).sum())


class logisticmodel(object):
    def __init__(self):
		pass
	#propogate forward and calculate the err	
    def forward(self,X):
        return (softmax(X.dot(self.W) + self.b))

    def get_error_rate(original,predictions):
        return(np.mean(original != predictions))

    def get_predictions(self,X):
        fY = self.forward(X)
        return(np.argmax(fY,axis=1))
    def find_score(self,X,Y):
        predictions = get_predictions(X)
        return(1- get_error_rate(Y,predictions))
    # Soft max fuction - the result at each ste
    def fit(self, X, Y, learning_rate=1e-7, reg=0., epochs=10000, show_fig=False):
        X, Y = shuffle(X, Y)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        Tvalid = y2indicator(Yvalid)
        X, Y = X[:-1000], Y[:-1000]

        N, D = X.shape
        K = len(set(Y))
        T = y2indicator(Y)
        self.W = np.random.randn(N, D) / np.sqrt(D)
        self.b = np.zeros(K)

        costs = []
        best_validation_error = 1
        for i in range(epochs):
            # forward propagation and cost calculation
            pY = self.forward(X)

            # gradient descent step
            self.W -= learning_rate*(X.T.dot(pY - T) + reg*self.W)
            self.b -= learning_rate*((pY - T).sum(axis=0) + reg*self.b)

            if i % 10 == 0:
                pYvalid = self.forward(Xvalid)
                c = cost(Tvalid, pYvalid)
                costs.append(c)
                e = error_rate(Yvalid, np.argmax(pYvalid, axis=1))
                print("i:", i, "cost:", c, "error:", e)
                if e < best_validation_error:
                    best_validation_error = e
        print("best_validation_error:", best_validation_error)

        if show_fig:
            plt.plot(costs)
            plt.show()



def main():
	X, Y = getdata()
	model = logisticmodel()

	model.fit(X,Y)
	print(model.find_score(X,Y))
if __name__ == "__main__":
	main()



