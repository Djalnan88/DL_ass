import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time
from dataset_PA1.dataloader import Dataloader
from dataset_PA1.dataloader import datasetIterator

random.seed(42)

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    H_out = (H + 2 * pad - filter_h) // stride + 1
    W_out = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')

    col = np.zeros((N, C, filter_h, filter_w, H_out, W_out))

    for y in range(filter_h):
        y_max = y + stride * H_out
        for x in range(filter_w):
            x_max = x + stride * W_out
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * H_out * W_out, -1)
    return col, H_out, W_out

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    H_padded, W_padded = H + 2 * pad, W + 2 * pad

    H_out = (H + 2 * pad - filter_h) // stride + 1
    W_out = (W + 2 * pad - filter_w) // stride + 1
    col = col.T.reshape(N, H_out, W_out, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H_padded, W_padded), dtype=col.dtype)

    for y in range(filter_h):
        y_max = y + stride * H_out
        for x in range(filter_w):
            x_max = x + stride * W_out
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    if pad == 0:
        return img
    else:
        return img[:, :, pad:H_padded - pad, pad:W_padded - pad]

class Conv2DLayer:
    def __init__(self, input_shape, out_channels, kernel_size, stride=1, padding=0, learning_rate=0.01):
        N, C_in, H_in, W_in = input_shape
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.learning_rate = learning_rate

        self.weights = np.random.randn(out_channels, C_in, kernel_size, kernel_size) * 0.01
        self.biases = np.zeros(out_channels)
        
        self.input_shape = input_shape
        self.batch_size = N
        self.col = None
        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)
        self.H_out = None
        self.W_out = None
    
    def forward(self, x):
        N, C, H, W = x.shape
        K, P, S = self.kernel_size, self.padding, self.stride

        col, H_out, W_out = im2col(x, K, K, S, P)
        self.col = col
        self.H_out = H_out
        self.W_out = W_out

        W_col = self.weights.reshape(self.out_channels, -1)
        out_matrix = np.dot(col, W_col.T) + self.biases[np.newaxis, :]

        output = out_matrix.T.reshape(N, H_out, W_out, self.out_channels).transpose(0, 3, 1, 2)
        return output
    
    def backward(self, dvalues):
        N, C_out, H_out, W_out = dvalues.shape
        K, P, S = self.kernel_size, self.padding, self.stride

        self.dbiases = np.sum(dvalues, axis=(0, 2, 3)) / self.batch_size
        dvalues_col = dvalues.transpose(0, 2, 3, 1).reshape(N*H_out*W_out, C_out)

        dweights_col = np.dot(self.col.T, dvalues_col)
        self.dweights = dweights_col.reshape(self.weights.shape) / self.batch_size

        W_col = self.weights.reshape(C_out, -1)
        dinput_col = np.dot(dvalues_col, W_col)
        dinput = col2im(dinput_col, self.input_shape, K, K, S, P)

        return dinput
    
    def update_params(self):
        self.weights -= self.learning_rate * self.dweights
        self.biases -= self.learning_rate * self.dbiases

class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
        self.input = None
    
    def forward(self, x):
        N, C, H_int, W_int = x.shape
        self.input = x

        K = self.pool_size
        S = self.stride

        H_out = (H_int - K) // S + 1
        W_out = (W_int - K) // S + 1

        output = np.zeros((N, C, H_out, W_out))

        self.mask = np.zeros_like(x)

        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * S
                        h_end = h_start + K
                        w_start = w * S
                        w_end = w_start + K

                        window = x[n, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(window)
                        output[n, c, h, w] = max_val

                        max_h, max_w = np.unravel_index(np.argmax(window), window.shape)
                        self.mask[n, c, h_start + max_h, w_start + max_w] = 1
        return output
    
    def backward(self, dvalues):
        dinput = np.zeros_like(self.input)
        N, C, H_out, W_out = dvalues.shape
        K = self.pool_size
        S = self.stride

        for n in range(N):
            for c in range(C):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * S
                        h_end = h_start + K
                        w_start = w * S
                        w_end = w_start + K

                        mask_window = self.mask[n, c, h_start:h_end, w_start:w_end]
                        dinput[n, c, h_start:h_end, w_start:w_end] += mask_window * dvalues[n, c, h, w]
        return dinput
    
class LinearLayer:
    def __init__(self, batch_size, input_size, output_size, learning_rate=0.001):
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        limit = np.sqrt(6 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        self.biases = np.zeros((1, output_size))

        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)

    def forward(self, x):
        self.input = x
        output = np.dot(x, self.weights) + self.biases
        return output

    def backward(self, dvalues):
        self.dinput = np.dot(dvalues, self.weights.T)
        self.dweights = np.dot(self.input.T, dvalues) / self.batch_size
        self.dbiases = np.sum(dvalues, axis=0) / self.batch_size
        return self.dinput
    
    def update_params(self):
        self.weights -= self.learning_rate * self.dweights
        self.biases -= self.learning_rate * self.dbiases

class ReLULayer:
    def __init__(self):
        pass
    
    def forward(self, x):
        self.input = x
        output = np.maximum(0, x)
        return output
    
    def backward(self, dvalues):
        self.dinput = dvalues.copy()
        self.dinput[self.input < 0] = 0
        return self.dinput

def softmax(x):
    exp_real = np.exp(x - np.max(x, axis=1, keepdims=True))
    pred = exp_real / np.sum(exp_real, axis=1, keepdims=True)
    return pred

def cross_entropy_loss(pred, ans):
    n = pred.shape[0]
    log_prob = np.log(pred[range(n), ans])
    loss = -np.sum(log_prob) / n
    return loss

def softmax_cross_entropy_backward(pred, ans):
    n = pred.shape[0]
    grad = pred - np.eye(pred.shape[1])[ans]
    grad = grad / n
    return grad