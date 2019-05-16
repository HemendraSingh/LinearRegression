import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle

def main():
    train_model()
    predictUsingModel()
    
def train_model():
    x = np.array([2014, 2015, 2016, 2017, 2018]).reshape((-1, 1))
    y = np.array([200, 260, 230, 250, 300])

    model = LinearRegression()
    model.fit(x,y)

    r_sq = model.score(x, y)
    print('coefficient of determination:', r_sq)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)
    print("Linear Regression Equation: Y = ",model.intercept_," + ",model.coef_[0], "X",sep='')

    y_pred = model.predict(x)
    print('predicted response:', y_pred, sep='\n')
    
    filename = 'lrm_model.sav'
    pickle.dump(model, open(filename, 'wb'))

def predictUsingModel():
    x = np.array([2014, 2015, 2016, 2017, 2018]).reshape((-1, 1))
    y = np.array([200, 260, 230, 250, 300])

    model = pickle.load(open("lrm_model.sav", 'rb'))
    r_sq = model.score(x, y)
    print('coefficient of determination:', r_sq)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)
    print("Linear Regression Equation: Y = ",model.intercept_," + ",model.coef_[0], "X",sep='')

    y_pred = model.predict(x)
    print('predicted response:', y_pred, sep='\n')
    
    plt.scatter(x,y,color='green')
    plt.scatter(x,y_pred,color='black')
    plt.plot(x,y_pred,color='black',linewidth=3)
    plt.show()

if __name__ == "__main__":
    main()