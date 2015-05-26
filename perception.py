# -*- coding: utf-8 -*-
#@author: glay8717@gmail.com
#@perception algorithm
from numpy import *
import matplotlib.pyplot as plt
class singal_perception():
    w = 0
    b = 0
    n = 0.1
    it = 1000
    
    def __init__(self):
        print 'singal_perception init'

    def output(self, input_sample):
        x = sum(input_sample * self.w) + self.b
        return sign(x + 0.0001)

    def loss(self, input_data, input_label):
        num_samples = input_data.shape[0]
        l = 0
        for i in range(num_samples):
            if self.output(input_data[i]) * input_label[i] < 0:
                #print 'mistake point:', input_data[i]
                l -= (sum(input_data[i] * self.w) + self.b) * input_label[i]
        return l
            
    def stochastic_gradient_descent(self, input_data, input_label):
        mistake_sample = []
        num_samples = input_data.shape[0]
        for i in range(num_samples):
            if self.output(input_data[i]) * input_label[i] < 0:
                mistake_sample.append(i)
        
        num_mistake = len(mistake_sample)
        if num_mistake > 0:
            idx = mistake_sample[randint(num_mistake)]
            self.w += input_data[idx] * input_label[idx] * self.n
            self.b += input_label[idx] * self.n
        return num_mistake
        
    def train(self, input_data, input_label):
        print 'iteration | loss      | mistake:'
        for i in range(self.it):
            l = self.loss(input_data, input_label)
            num_mistake = self.stochastic_gradient_descent(input_data, input_label)
            print '%9d | %.7f | %d'  %(i, l, num_mistake)
            self.draw(input_data, input_label)
            if num_mistake % input_data.shape[0] == 0:
                break
    
        print 'The perception training finished :', self.w, self.b
        
    def draw(self, input_data, input_label):
        num_posi = count_nonzero(input_label == 1.0)
        x = linspace(0, 1, 100)
        y = -(self.w[0] * x + self.b)/self.w[1]
        plt.cla()
        # x is positive，o is negative
        plt.plot(input_data[:num_posi, 0], input_data[:num_posi, 1], 'x')
        plt.plot(input_data[num_posi:,0], input_data[num_posi:, 1],'o')        
        plt.plot(x, y)
        plt.show()
        plt.pause(0.1)
        
        
    def test(self):
        num_posi = 100
        num_nega = 100
        data = rand(num_posi + num_nega, 2)
        label = ones(num_posi + num_nega)
        data[:num_posi, :] = uniform(0, 0.5, (num_posi, 2))
        data[num_posi:, :] = uniform(0.5, 1.0, (num_nega, 2))
        label[:num_posi] = -1.0
        
        self.w = ones(data.shape[1])
        #print 'Traning data : \n', data
        #self.draw(data)
        self.train(data, label)
       
def main():
    handle = singal_perception()
    handle.test()
    
if __name__ == "__main__":
    main()
    