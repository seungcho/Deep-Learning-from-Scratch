#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict


# In[2]:


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)
        
        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'],self.params['b1']) # 입력값: W(가중치), b(편향)
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'],self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()
        
        # 예측 정의
        # 순전파를 통한 예측 값 출력
    def predict(self,x):
        for layer in self.layers.values():
            x = layer.forward(x) # 입력값: x(data point), np.dot(self.X, self.W) + self.b
            
        return x
        
        # 손실 함수 정의
        # 예측 값과 실제 값에 대한 손실 함수 값 출력 / 정확도 측정
    def loss(self,x,t):
        y = self.predict(x) 
        return self.lastLayer.forward(y,t)
        
    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y==t) / float(x.shape[0]) 
        
        return accuracy
        
        # 현재 손실함수에서의 기울기 출력
    def numerical_gradient(self,x,t):
        loss_W = lambda W: self.loss(x,t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W,self.params['W1']) # 입력 값: f(함수), x(가중치)
        grads['b1'] = numerical_gradient(loss_W,self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W,self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W,self.params['b2'])
        
        return grads
        
        # 매 epochs 별 손실함수에 대한 정보 저장 및 역전파
    def gradient(self,x,t):
        # 순전파
        self.loss(x,t)
        
        # 역전파
        dout = 1
        dout = self.lastLayer.backward(dout) # SoftmaxWithLoss의 역전파 즉, (self.y - self.t) / batch_size
        
        # 역전파를 위한 기존 입력 값들을 reverse 시켜줌
        layers = list(self.layers.values())
        layers.reverse() 
        
        for layer in layers:
            dout = layer.backward(dout)
            
        # 결과저장
        # loss 값에 대한 각 가중치의 기울기 값들을 저장
        grads = {}
        grads['W1'] = self.layer['Affine1'].dW
        grads['b1'] = self.layer['Affine1'].db
        grads['W2'] = self.layer['Affine2'].dW
        grads['b2'] = self.layer['Affine2'].db
        
        return grads

# In[ ]:




