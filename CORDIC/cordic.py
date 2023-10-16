 #!/usr/bin/env python3 
# -*- coding: utf-8 -*- 

#================================================
# Author : Rohan Panicker
# Created Date: 12/4/22
# version ='1.0'
#================================================

import sys
sys.path.append(r"C:\Users\HP\GitFiles\Neural-Networks")

import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt
from pathlib import Path as p
import XMLreader



class nn:
    def __init__(self, memory, alpha, optimizer, iterations, counter, PIDflag, Loss):
        self.memory = p(memory)
        self.weights, self.biases, self.acc = [], [], []
        self.alpha = alpha
        self.LearnFlag = True
        self.iter = iterations
        self.optimizer = optimizer
        self.LossFn = Loss
        self.loss_cache = np.array([])
        self.PIDflag = PIDflag
        self.counter = counter
        if (PIDflag):
             self.kp = 1.5; self.ki = 0.4; self.kd = 0.007; self.error_prev = self.error_integ = 0.

    def put(self, inp, output):
        self.input = inp
        self.output = output
        

    def architecture(self, growflag, flag, Activations, *layers):
        self.growflag = growflag
        self.layers = list(layers)
        self.Activations = Activations
        if growflag:
             self.DDwCache = [0.0 for _ in range(len(self.layers))]
             self.DDbCache = [0.0 for _ in range(len(self.layers))]
        if self.optimizer == "momentum":
            self.mu = 0.99
            self.vw, self.vb = [], []
            for l in range(len(self.layers)-1):
                self.vw.append(np.zeros((self.layers[l+1],  self.layers[l]))) 
                self.vb.append(np.zeros((self.layers[l+1],  1)))
                
        if flag:
            self.RandomizeWb()
            

    def RandomizeWb(self):
        with open(self.memory,'w') as d:
                for l in range(len(self.layers)-1):
                    for i in range(self.layers[l+1]):
                        for j in range(self.layers[l] + 1):
                            d.write(str(np.random.randn() + random.choice([-2, -1, 0, 1, 2]) )+", ")
                        d.write('\n')

    
    # def weightInspection(self):
    #      for i in range(len(self.layers) - 1):
                 
                 
    #      print("New Architecture : \n", self.layers)
        
    def run(self):
        print("Begin..")
        counter = 0
        if (self.LearnFlag == False):
             self.LearnFlag = True
        while (self.LearnFlag):
            if (counter > self.counter):
                 print("Unable to Learn!!!!!")
                 break;
            self.reader()
            self.forprop()
            self.error()
            print("Error: {}".format(self.loss))
            print("-------------")
            self.backprop()
            self.learn()
            self.writer()
            counter+=1
    
    def predict(self):
        self.reader()
        self.forprop()
        
        print("actual and predicted are:\n",self.output,'\n and \n',self.output_)
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
#        self.weightInspection();

    def writer(self):
        with open(self.memory, 'w') as d:
            for ws, bs in zip(self.weights, self.biases):
                for w,b in zip(ws,bs):
                    for k in w:
                        d.write(str(k)+", ")
                    d.write(str(b[0])+'\n')

    def PID(self):
        err = self.loss;
        self.error_integ += err;
        err_der = err - self.error_prev
        self.alpha*= (self.kp*err + self.ki*self.error_integ + self.kd*err_der)
         

    def reshuffle(self):
         print("Reshuffled!!")
         self.RandomizeWb()
         self.reader()

    def error(self):
         getattr(self, self.LossFn)()
         self.loss_cache = np.append(self.loss_cache, self.loss)
         if self.loss < 0.01:
                   self.LearnFlag = False
         if len(self.loss_cache)>=10:
              if abs(self.loss_cache[-1] - self.loss_cache[-2]) <1e-2 and\
                 abs(self.loss_cache[-2] - self.loss_cache[-3]) <1e-2 and\
                 abs(self.loss_cache[-3] - self.loss_cache[-4]) <1e-2 and\
                 abs(self.loss_cache[-4] - self.loss_cache[-5]) <1e-2:
                   self.loss_cache = np.array([])
                   self.reshuffle();
              else:   
                 self.loss_cache = np.delete(self.loss_cache, -1)
         if self.PIDflag:
                    self.loss_cache = np.append(self.loss_cache, self.loss)
                    self.PID()


    def cross(self):
        self.loss = self.output*np.log(self.output_) + (1-self.output)*np.log(1-self.output_)*(1/len(self.output_))

    def cross_der(self):
         setattr(self, "dA"+str(len(self.layers)-1), - (np.divide(self.output, self.output_) - np.divide(1 - self.output, 1 - self.output_)))
         
    def L2(self):
         self.loss = np.square(self.output_ - self.output).mean()
  
    def L2_der(self):
         setattr(self, "dA"+str(len(self.layers)-1),  2 * np.mean(self.output_ - self.output) )
         

    def L1(self):
         self.loss = np.mean(self.output_ - self.output)

    def L1_der(self):
         setattr(self, "dA"+str(len(self.layers)-1), self.output)

    def Huber(self):
         if (Hu:=self.L1()) <= 0.1:
              self.L2();
         self.loss = Hu
    

    def filter(self, inp, kernel, stride, pad, func):   #   incase needed, especially for cnn
            n = 1+ (len(inp) - len(kernel) + 2*pad)//stride
            output = np.zeros((n,n))

            for r in range(0, n,stride):
                for c in range(0, n, stride):
                    output[r,c]  = func(np.multiply(inp[r:r+len(kernel), c:c+len(kernel)], kernel))
            return output


    def forprop(self):
        '''
        z = wx + b
        a = g(z)
        '''

        self.inp = self.a0 =  self.input
        for idx, (layer, bias) in enumerate(zip(self.weights, self.biases),1):
            setattr(self, 'z'+str(idx), np.dot(layer, self.inp) + bias)
            setattr(self, 'a'+str(idx), getattr(self, self.Activations[idx - 1])(np.array(getattr(self, 'z'+str(idx)), dtype = float)))
            self.inp = getattr(self, 'a'+str(idx))
        self.output_ = getattr(self, 'a'+str(len(self.layers)-1))


    def polyforprop(self):
        '''
        z = vx^2 + wx + b
        a = g(z) 
        '''
        pass

    def polybackprop(self):
        '''
        tbd
        '''
        pass


        
    def backprop(self):
        '''
        da[l-1] = W[l].T x dz[l]
        dz[l-1] = da[l-1] x g'(a[l-1])
        dw[l-1] = dz[l-1] a[l-2]/m



        '''
        getattr(self, self.LossFn + "_der")()
        # setattr(self, "dA"+str(len(self.layers)-1), - (np.divide(self.output, self.output_) - np.divide(1 - self.output, 1 - self.output_)))
        
        for i in range(len(self.layers)-1, 0, -1):
            if i<len(self.layers)-1:
                setattr(self, "dA"+str(i), np.dot(self.weights[i].T, getattr(self, "dZ"+str(i+1))))
            setattr(self, "dZ"+str(i), getattr(self, "dA"+str(i)) * getattr(self, self.Activations[i - 1] + "_der")(getattr(self, "a"+str(i))))
            
            z = getattr(self, "dZ"+str(i))
            a = getattr(self, 'a'+str(i-1))
            setattr(self, "dW"+str(i), np.dot(z, a.T)/a.shape[1])
            setattr(self, "dB"+str(i), np.sum(z, axis = 1, keepdims = True)/a.shape[1])
        
        if self.growflag:
            '''
            printing self.dW[i] will print all the differential weights of that layer.
            So the plan is to:

            Have the loss of the current cycle,
            Check individual differential weights of each layer.

            The crux is to find a relation between the loss and the differential weights such
            that we could either remove those weights or add more.

            This is for resizing the neural size for better performance.
             
            '''
            for i in range(1, len(self.layers)-1):
                  self.DDwCache[i]  = (getattr(self, "dW"+str(i))) - self.DDwCache[i]
                  self.DDbCache[i]  = (getattr(self, "dB"+str(i))) - self.DDbCache[i]
                  print(self.DDwCache[i], 'd');

    def learn(self):
         getattr(self, self.optimizer)()
    
    def vanilla(self):
        '''
        w = w - @dw
        b = b - @db
        '''
        for i in range(len(self.layers)-1):
                self.weights[i]-=self.alpha*getattr(self, "dW"+str(i+1))
                self.biases[i] -=self.alpha*getattr(self, "dB"+str(i+1))

    def momentum(self):
        for i in range(len(self.layers)-1):
            assert(self.vw[i].shape == self.weights[i].shape)
            self.vw[i] = self.mu*self.vw[i] - self.alpha*getattr(self, "dW"+str(i+1))
            self.weights[i]+=self.vw[i]
            self.vb[i] = self.mu*self.vb[i] - self.alpha*getattr(self, "dB"+str(i+1))
            self.biases[i] += self.vb[i]

    def linear(self, z):
        return z

    def linear_der(self, linz):
        return 1
    
    def RelU(self, z):
        return np.maximum(0, z)
    
    def RelU_der(self,relz):
        return np.where(relz > 0, 1, 0)
    
    def LeakyRelU(self, z):
        return np.maximum(np.dot(z, 0.1), z)
    
    def LeakyRelU_der(self, leakyrelz):
        return np.where(leakyrelz >= 0, 1, 0)

    def sigmoid(self,z):
        return 1/(1+np.exp(-z)) - 0.5

    def sigmoid_der(self,sigmz):
        return sigmz*(1-sigmz)
    
    def tanh(self, z):
        en  = np.exp(z)
        en_ = np.exp(-z) + 1e-6 
        return (en - en_)/(en + en_)
    
    def tanh_der(self,tanhz):
        return 1 - tanhz*tanhz

    def ntanh(self,z):
        return - self.tanh(z)

    def ntanh_der(self,z):
        return - self.tanh_der(z)
    
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

if __name__ == "__main__":
    from math import sin, pi

    filename = 'Data.xml'
    dom = XMLreader.xml.dom.minidom.parse(filename)
    NN = dom.getElementsByTagName("Neural-Networks_Parameter")
    XMLreader.ParseXML(NN)
    
    path   = ''.join(XMLreader.d["Path"])
    layers = (''.join(XMLreader.d["Layers"])).split(',')
    alpha =  float((XMLreader.d["Alpha"]))
    iterations = int(XMLreader.d["Epoch"])
    counter = int(XMLreader.d["counter"])
    trainflag = int(XMLreader.d["Train?"])
    growflag = int(XMLreader.d["Grow?"])
    Activations =  (''.join(XMLreader.d["Activation Functions"])).split(',')
    Optimizer = ''.join(XMLreader.d["Optimizer"])
    Loss    = ''.join(XMLreader.d["Loss Function"])
    PIDflag = int(XMLreader.d["PID"])

    print(Activations)
    if (len(Activations) != len(layers) - 1):
        print("Incorrect Config")
        exit();
    
    X = nn(path, alpha , Optimizer, iterations, counter, PIDflag, Loss);
    
    #CORDIC
    '''
    A neural network with the configuration (2, 4, 2, 1) where 1 is the output
    and 2 is the input, and 4, 2 are the hidden layers

    The learning rate, cycle, and iterations are 0.5, 40, 40

    '''
    def train(cycle):
        X.architecture(growflag, 1, Activations, *(int(i) for i in layers)) #(flag, *layers)
        plt.ion()
        plt.grid()
        a = 0.00; 
        b = 2*pi
        for _ in range(cycle):
            if _%2:
                if a > 2*pi:
                     a = 0.0
                input_rad = a;
                output_sin = sin(input_rad)
                X.put(np.array([[input_rad]]),  np.array([[output_sin]]))
                X.run()
                a+=0.033333
            else:
                if b < 0.00:
                     b = 2*pi
                input_rad = b;
                output_sin = sin(input_rad)
                X.put(np.array([[input_rad]]),  np.array([[output_sin]]))
                X.run()
                b-=0.0133333

                # if _%3:
                #     for i in np.arange(pi/4, pi/2, 0.7):
                #                     input_rad = i;
                #                     output_sin = sin(input_rad)
                #                     X.put(np.array([[input_rad]]),  np.array([[output_sin]]))
                #                     X.run()
                # if _%5:
                #     for i in np.arange(pi/2, 3*pi/4, 0.7):
                #                     input_rad = i;
                #                     output_sin = sin(input_rad)
                #                     X.put(np.array([[input_rad]]),  np.array([[output_sin]]))
                #                     X.run()
                
                # if _%7:
                #     for i in np.arange(3*pi/4,pi, 0.7):
                #                     input_rad = i;
                #                     output_sin = sin(input_rad)
                #                     X.put(np.array([[input_rad]]),  np.array([[output_sin]]))
                #                     X.run()
                # if _%2:
                #     for i in np.arange(pi,5*pi/4, 0.7):
                #                     input_rad = i;
                #                     output_sin = sin(input_rad)
                #                     X.put(np.array([[input_rad]]),  np.array([[output_sin]]))
                #                     X.run()
                
                # if _%3:
                #     for i in np.arange(5*pi/4, 3*pi/2 ,0.7):
                #                     input_rad = i;
                #                     output_sin = sin(input_rad)
                #                     X.put(np.array([[input_rad]]),  np.array([[output_sin]]))
                #                     X.run()
                
                # if _%5:
                #     for i in np.arange(3*pi/2 ,7*pi/4,0.7):
                #                     input_rad = i;
                #                     output_sin = sin(input_rad)
                #                     X.put(np.array([[input_rad]]),  np.array([[output_sin]]))
                #                     X.run()
                # if _%7:
                #     for i in np.arange(7*pi/4, 2*pi, 0.7):
                #                     input_rad = i;
                #                     output_sin = sin(input_rad)
                #                     X.put(np.array([[input_rad]]),  np.array([[output_sin]]))
                #                     X.run()

            Nsx, sx = [], []
            for i in np.arange(0, 2*pi, 0.01):
                                        input_rad = i;
                                        output_sin = sin(input_rad)
                                        X.put(np.array([[input_rad]]),  np.array([[output_sin]]))
                                        X.predict()
                                        Nsx.append(X.output_[0][0])
                                        sx.append(output_sin)
            plt.clf()
            plt.plot(range(len(sx)), sx, 'r', range(len(Nsx)), Nsx, 'b');
            plt.show()
            plt.pause(0.1)
                    
        # for _ in range(cycle):
                    # input_rad = random.uniform(0, pi/2);
                    # output_sin = sin(input_rad)
                    # X.put(np.array([[input_rad]]),  np.array([[output_sin]]))
                    # X.run()

    def pred(cycle):
         X.architecture(growflag, 0, Activations, *(int(i) for i in layers)) #(flag, *layers)
         for _ in range(cycle):
            
            input_rad = random.uniform(0, pi/2);
            output_sin = sin(input_rad)

            print("\n theta: {}  Sin(Theta): {} \n".format(input_rad, output_sin))
            X.put(np.array([[input_rad]]),  np.array([[output_sin]]))
            X.predict()
           
         print("accuracy is ", 1-max(X.acc))
    
    def plot():
        
        X.architecture(growflag, 0, Activations, *(int(i) for i in layers)) #(flag, *layers)
        Nsx, sx = [], []
        for i in np.arange(0, 2*pi, 0.2):
                                    input_rad = i;
                                    output_sin = sin(input_rad)
                                    X.put(np.array([[input_rad]]),  np.array([[output_sin]]))
                                    X.predict()
                                    Nsx.append(X.output_[0][0])
                                    sx.append(output_sin)

        plt.plot(range(len(sx)), sx, 'r', range(len(Nsx)), Nsx, 'b');
        plt.show()

    if trainflag:
        train(iterations) 
    else:  
        #pred(40)
        plot()

