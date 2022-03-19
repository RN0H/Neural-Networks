

import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path as p


def randomweights():
    d = p(r"/path/weights.txt")

    with open(d,'w') as d:
        for _ in range(10):
            for _ in range(101):
                     d.write(str(np.random.randn())+', ')
            d.write('\n')

        for _ in range(5):
            for _ in range(11):
                     d.write(str(np.random.randn())+', ')
            d.write('\n')

        for _ in range(3):
            for _ in range(6):
                     d.write(str(np.random.randn())+', ')
            d.write('\n')


class nn:
    def __init__(self, image, output, memory):
        self.image = image
        self.output = output
        self.memory = p(memory)
        self.alpha = 0.001
        
    def run(self):
        print("Begin..")
        #y = np.arange(-0.2, 0.2,0.01)
        for _ in range(10000):
            self.reader()
            self.forprop()
            a = self.check()
            if -0.1<a[0]<0.1 and -0.1<a[1]<0.1 and -0.1<a[2]<0.1:
                print("Learnt")
                break
            self.backprop()
            self.learn()
            self.writer()
    
    def predict(self):
        self.reader()
        self.forprop()
        self.answer = self.output_
        print("actual and predicted are:",self.output,'\n and \n',self.answer)


    def reader(self):

        with open(self.memory,'r') as x:
            f= x.readlines()
            self.layer1, self.b1 =  np.array([[float(j) for j in i.split(', ')[:100]] for i in f[:10]], dtype = object), np.array([[float(i.split(', ')[100])] for i in f[:10]], dtype = object)
            self.layer2, self.b2 =  np.array([[float(j) for j in i.split(', ')[:10]] for i in f[10:15]], dtype = object), np.array([[float(i.split(', ')[10])] for i in f[10:15]], dtype = object)
            self.layer3, self.b3 =  np.array([[float(j) for j in i.split(', ')[:5]] for i in f[15:]], dtype = object), np.array([[float(i.split(', ')[5])] for i in f[15:]], dtype = object)


    def writer(self):

        with open(self.memory, 'w') as d:
            for be, we in zip(self.layer1,self.b1):
                for w in  we:
                    d.write(str(w)+', ')
                for b in be:
                    d.write(str(b)+', ')
                d.write('\n')

            for be, we in zip(self.layer2,self.b2):
                for w in  we:
                    d.write(str(w)+', ')
                for b in be:
                    d.write(str(b)+', ')
                d.write('\n')

            for be, we in zip(self.layer3,self.b3):
                for w in  we:
                    d.write(str(w)+', ')
                for b in be:
                    d.write(str(b)+', ')
                d.write('\n')
                

    def check(self):
        self.MSE = self.output - self.output_  #-self.output*np.log(self.output_) + (1-self.output)*np.log(1-self.output_)
        print(self.MSE)
        return self.MSE
        #self.MSE = np.mean( (self.output - self.output_), axis = 1)
        #print(self.MSE,'\n',self.output)
        #self.MSE = np.array([list(self.MSE) for _ in range(10)]).reshape(3,10)
        




    def forprop(self):
        self.z1 =  np.dot(self.layer1, self.image) + self.b1 #10x100, 100x1
        assert self.z1.shape, (10, 1)
        self.a1 = self.sigmoid(np.array(self.z1, dtype = float))

        self.z2 =  np.dot(self.layer2, self.a1) + self.b2 #5x10, 10x1
        assert self.z2.shape, (5,1)
        self.a2 = self.sigmoid(np.array(self.z2, dtype = float))

        self.z3 = np.dot(self.layer3, self.a2) + self.b3  #3x5, 5x1
        assert self.z3.shape, (3,1)
        self.a3= self.softmax(np.array(self.z3, dtype = float))
        self.output_ = self.a3
                


    def backprop(self):
        
        self.e =  self.MSE                                  #3x1
        self.d =  np.dot(self.e.T, self.softmax_der(self.output_))   #1x3 3x3  = 1x3

        self.e3 = np.dot(self.layer3.T, self.d.T)             #5x3 3x1
        self.dW3 = self.e3 * self.sigmoid_der(self.a2)      #5x1
        self.db3 = np.mean(self.e3)*np.ones(self.b3.shape)
        
        self.e2 = np.dot(self.layer2.T,self.dW3)              #10x5 5x1
        self.dW2 = self.e2 * self.sigmoid_der(self.a1)        #10x1
        self.db2= np.mean(self.e2)*np.ones(self.b2.shape)

        self.e1 = np.dot(self.layer1.T, self.dW2)             #100x10 10x1
        self.dW1 = self.e1 * self.sigmoid_der(self.image)        #100x1  
        self.db1 = np.mean(self.e1)*np.ones(self.b1.shape)
        

    def learn(self):

        self.layer1-=self.alpha*self.a1.dot(self.dW1.T)      #100, 100x10   (10x10)
        self.layer2-=self.alpha*self.a2.dot(self.dW2.T)      #5x10 10x10     (5x10)
        self.layer3-=self.alpha*self.a3.dot(self.dW3.T)      #3x10, 10x5     (3x5)

        self.b1-=self.alpha*self.db1
        self.b2-=self.alpha*self.db2
        self.b3-=self.alpha*self.db3

    def softmax(self, z):
        return np.exp(z)/np.sum(np.exp(z))

    def softmax_der(self, z):
        soft_der = np.zeros((len(z), len(z)))
        for j in range(len(z)):
            for i in range(len(z)):
                if i == j:
                    soft_der[i,i] = z[i]*(1-z[i])
                else:
                    soft_der[i,j] = -z[i]*z[j]
        return soft_der

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))

    def sigmoid_der(self,z):
        return z*(1-z)

if __name__ == "__main__":
    randomweights()

    def train(n):
        C = np.zeros(100).reshape(10,10)
        a,b,c = 0,0,0
        for epoch in range(n):
            choice = random.randint(0,11)/10
            if choice<0.3333:
                x,y = random.randint(1,6),random.randint(1,6)
                C[x,y] = C[x,y+1] = C[x, y+2] =  C[x+1,y+2] =  C[x+2, y+2] = C[x+2,y+1] = C[x+2, y] = C[x+1, y] = 1
                nn(C.reshape(100,1),np.array([[1], [0], [0]]),"/home/rohan/projects/3py/weights.txt").run()
                a+=1
            elif 0.3333<choice<0.6667:
                I = C.copy()
                x = random.randint(3,7)
                I[:,x] = 1
                nn(I.reshape(100,1),np.array([[0], [1], [0]]),"/home/rohan/projects/3py/weights.txt").run()
                b+=1
            else:
                x,y = random.randint(1,9),random.randint(1,9)
                X = C.copy()
                X[:,y] = 1
                X[x,:] =1 
                nn(X.reshape(100,1),np.array([[0], [0], [1]]),"/home/rohan/projects/3py/weights.txt").run()
                c+=1
        print(a,b,c)

    def pred():
        C = np.zeros(100).reshape(10,10)
        for epoch in range(100):
            choice = random.randint(1,10)/10
            if  choice< 0.3:    #O
                x,y = random.randint(3,7),random.randint(3,7)
                C[x,y] = C[x,y+1] = C[x, y+2] =  C[x+1,y+2] =  C[x+2, y+2] = C[x+2,y+1] = C[x+2, y] = C[x+1, y] = 1
                nn(C.reshape(100,1),np.array([[1], [0], [0]]),"/home/rohan/projects/3py/weights.txt").predict()
                    
            elif 0.3< choice <0.7:  #I
                I = C.copy()
                x = random.randint(3,7)
                I[:,x] = 1
                nn(I.reshape(100,1),np.array([[0], [1], [0]]),"/home/rohan/projects/3py/weights.txt").predict()
                
            else: #X
                x,y = random.randint(3,7),random.randint(3,7)
                X = C.copy()
                X[:,y] = 1
                X[x,:] =1
                nn(X.reshape(100,1),np.array([[0], [0], [1]]),"/home/rohan/projects/3py/weights.txt").predict()
    
    
    train(100)
    pred()