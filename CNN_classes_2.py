import numpy as np

def im2col(x, k):
    batch, c, h, w = x.shape
    out_h, out_w = h - k + 1, w - k + 1
    cols = np.zeros((batch, c * k * k, out_h * out_w))
    for i in range(k):
        for j in range(k):
            cols[:, (i*k+j)*c:(i*k+j+1)*c, :] = x[:, :, i:i+out_h, j:j+out_w].reshape(batch, c, -1)
    return cols

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, lr):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        limit = 1 / np.sqrt(in_channels * kernel_size * kernel_size)
        self.W = np.random.uniform(-limit, limit, (out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.zeros((out_channels, 1))
        self.lr = lr

    def forward(self, x):
        self.x = x
        batch, _, h, w = x.shape
        k = self.kernel_size
        out_h, out_w = h - k + 1, w - k + 1
        x_col = im2col(x, k).transpose(0, 2, 1)  # (batch, out_h*out_w, c_in*k*k)
        W_col = self.W.reshape(self.out_channels, -1)  # (out_channels, c_in*k*k)
        out = np.matmul(x_col, W_col.T) + self.b.T  # (batch, out_h*out_w, out_channels)
        out = out.transpose(0, 2, 1).reshape(batch, self.out_channels, out_h, out_w)
        return out

    def _conv2d_single(self, img, kernel):
        h, w = img.shape
        k = kernel.shape[0]
        out = np.zeros((h - k + 1, w - k + 1))
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                out[i, j] = np.sum(img[i:i+k, j:j+k] * kernel)
        return out

    def backward(self, d_out):
        batch, c_in, h, w = self.x.shape
        k = self.kernel_size
        d_x = np.zeros_like(self.x)
        d_W = np.zeros_like(self.W)
        d_b = np.zeros_like(self.b)

        for n in range(batch):
            for oc in range(self.out_channels):
                d_b[oc] += np.sum(d_out[n, oc])
                for ic in range(self.in_channels):
                    for i in range(h - k + 1):
                        for j in range(w - k + 1):
                            region = self.x[n, ic, i:i+k, j:j+k]
                            d_W[oc, ic] += region * d_out[n, oc, i, j]
                            d_x[n, ic, i:i+k, j:j+k] += self.W[oc, ic] * d_out[n, oc, i, j]

        # update
        self.W -= self.lr * d_W / batch
        self.b -= self.lr * d_b / batch

        return d_x

class ReLU:
    def forward(self, x):
        self.x = x
        self.mask = (x > 0)
        return np.maximum(0, x)

    def backward(self, d_out):
        return d_out * self.mask

class MaxPool2D:
    def __init__(self, size=2):
        self.size = size

    def forward(self, x):
        self.x = x
        b, c, h, w = x.shape
        s = self.size
        x_reshaped = x.reshape(b, c, h//s, s, w//s, s)
        out = x_reshaped.max(axis=(3, 5))
        self.max_mask = (x == out.repeat(s, axis=2).repeat(s, axis=3))
        return out

    def backward(self, d_out):
        b, c, h, w = self.x.shape
        s = self.size
        d_x = np.zeros_like(self.x)
        d_out_expanded = d_out.repeat(s, axis=2).repeat(s, axis=3)
        d_x[self.max_mask] = d_out_expanded[self.max_mask]
        return d_x

class Flatten:
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, d_out):
        return d_out.reshape(self.input_shape)

class Linear:
    def __init__(self, in_features, out_features, lr):
        limit = 1 / np.sqrt(in_features)
        self.W = np.random.uniform(-limit, limit, (in_features, out_features))
        self.b = np.zeros((1, out_features))
        self.lr = lr

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b
    
    def backward(self, d_out):
        dW = self.x.T @ d_out
        db = np.sum(d_out, axis=0, keepdims=True)
        dx = d_out @ self.W.T
        self.W -= self.lr * dW
        self.b -= self.lr * db
        return dx

class SoftmaxCrossEntropy:
    def forward(self, logits, labels):
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        self.probs = probs
        self.labels = labels
        loss = -np.sum(labels * np.log(probs + 1e-9)) / logits.shape[0]
        return loss

    def backward(self):
        return (self.probs - self.labels) / self.labels.shape[0]