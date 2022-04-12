 #!/usr/bin/env python3 
# -*- coding: utf-8 -*- 

#================================================
# Author : Rohan Panicker
# Created Date: 12/4/22
# version ='1.0'
#================================================


import numpy as np
import random
import matplotlib.pyplot as plt
from pathlib import Path as p


class nn:
    def __init__(self, memory, alpha, iterations):
        self.memory = p(memory)
        self.weights, self.biases, self.acc = [], [], []
        self.alpha = alpha
        self.iter = iterations


    def put(self, inp, output):
        self.input = inp
        self.output = output
        

    def architecture(self, flag, *layers):
        d = self.memory
        self.layers = layers
        if flag:
            with open(d,'w') as d:
                for l in range(len(self.layers)-1):
                    for i in range(self.layers[l+1]):
                        for j in range(self.layers[l] + 1):
                            d.write(str(np.random.randn())+", ")
                        d.write('\n')

        
    def run(self):
        print("Begin..")
        for _ in range(self.iter):
            self.reader()
            self.forprop()
            a = self.check()
            self.backprop()
            self.learn()
            self.writer()
    
    def predict(self):
        self.reader()
        self.forprop()
        
        print("actual and predicted are:",self.output,'\n and \n',self.output_)
        self.acc.append(self.output[0]-self.output_[0])
        print(self.acc[-1])


    def reader(self):                                               

        self.weights, self.biases = [], []                                                                            
        with open(self.memory,'r') as x:                                          
                                                                                                                                                          
            f= x.readlines()                                           
            st = 0
            for l in range(len(self.layers)-1):
                end = self.layers[l+1]                
                self.weights.append(np.array([[float(j) for j in i.split(', ')[:self.layers[l]]] for i in f[st: st + end] ], dtype = object))
                self.biases.append(np.array([[float(i.split(', ')[self.layers[l]])] for i in f[st: st+end]], dtype = object))

                st+=end

    def writer(self):
        with open(self.memory, 'w') as d:
            for ws, bs in zip(self.weights, self.biases):
                for w,b in zip(ws,bs):
                    for k in w:
                        d.write(str(k)+", ")
                    d.write(str(b[0])+'\n')


    def check(self):
        self.loss = self.output*np.log(self.output_) + (1-self.output)*np.log(1-self.output_)*(1/len(self.output_))
        print(self.output, '--',self.output_)
        print("-------------")
        return self.loss

        

    def filter(self, inp, kernel, stride, pad, func):   #   incase needed, especially for cnn
            n = 1+ (len(inp) - len(kernel) + 2*pad)//stride
            output = np.zeros((n,n))

            for r in range(0, n,stride):
                for c in range(0, n, stride):
                    output[r,c]  = func(np.multiply(inp[r:r+len(kernel), c:c+len(kernel)], kernel))
            return output


    def forprop(self):

        self.inp = self.a0 =  self.input
        for idx, (layer, bias) in enumerate(zip(self.weights, self.biases),1):
            setattr(self, 'z'+str(idx), np.dot(layer, self.inp)+bias)
            setattr(self, 'a'+str(idx), self.sigmoid(np.array(getattr(self, 'z'+str(idx)), dtype = float)))
            self.inp = getattr(self, 'a'+str(idx))

        self.output_ = getattr(self, 'a'+str(len(self.layers)-1))


    def backprop(self):
        '''
        da[l-1] = W[l].T x dz[l]
        dz[l-1] = da[l-1] x g'(a[l-1])
        dw[l-1] = d[l-1] a[l-2]/m
        '''
        setattr(self, "dA"+str(len(self.layers)-1), - (np.divide(self.output, self.output_) - np.divide(1 - self.output, 1 - self.output_)))
        
        for i in range(len(self.layers)-1, 0, -1):
            if i<len(self.layers)-1:
                setattr(self, "dA"+str(i), np.dot(self.weights[i].T, getattr(self, "dZ"+str(i+1))))
            setattr(self, "dZ"+str(i), getattr(self, "dA"+str(i))*self.sigmoid_der(getattr(self, "a"+str(i))))
            z = getattr(self, "dZ"+str(i))
            a = getattr(self, 'a'+str(i-1))
            setattr(self, "dW"+str(i), np.dot(z, a.T)/a.shape[1])
            setattr(self, "dB"+str(i), np.sum(z, axis = 1, keepdims = True)/a.shape[1])

    
    def learn(self):
        '''
        w = w - @dw
        b = b - @db
        '''
        
        for i in range(len(self.layers)-1):
            self.weights[i]-=self.alpha*getattr(self, "dW"+str(i+1))
            self.biases[i]-=self.alpha*getattr(self, "dB"+str(i+1))

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

    
    X = nn("/home/rohan/Neural_Networks/XOR/weights.txt", 0.1, 40);
    

    #XOR
    '''
    A neural network with the configuration (2, 4, 2, 1) where 1 is the output
    and 2 is the input, and 4, 2 are the hidden layers

    The learning rate, cycle, and iterations are 0.5, 40, 40

    '''

    def train(cycle):
        X.architecture(1, 2,4,2,1) #(flag, *layers)
        for _ in range(cycle):
            X.put(np.array([[0],[0]]), np.array([[0]]))
            X.run()

            X.put(np.array([[0],[1]]), np.array([[1]]))
            X.run()

            X.put(np.array([[1],[0]]), np.array([[1]]))
            X.run()

            X.put(np.array([[1],[1]]), np.array([[0]]))
            X.run()

    def pred(cycle):
         X.architecture(0, 2,4,2,1) #(flag, *layers)
         for _ in range(cycle):
            print('00')
            X.put(np.array([[0],[0]]), np.array([[0]]))
            X.predict()

            print('01')
            X.put(np.array([[0],[1]]), np.array([[1]]))
            X.predict()

            print('10')
            X.put(np.array([[1],[0]]), np.array([[1]]))
            X.predict()

            print('11')
            X.put(np.array([[1],[1]]), np.array([[0]]))
            X.predict()
         print("accuracy is ", 1-max(X.acc))
    
    #train(40)   
    pred(40)


