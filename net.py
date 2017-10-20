import numpy as np


class Layer(object):
  def __init__(self, params):
    pass

  def forward(self, inp):
    pass

  def backward(self, d_o):
    pass

  def update(self, eta):
    pass

  def test(self):
    pass


# Input is 5c dimensional, where c is set of all characters used in 3 languages 
class Encoder(Layer):
  def __init__(self, context, charset):
    self.context = context
    self.charset = charset
    self.size = len(charset)
    self.keys = [k for k in charset.keys()]
    self.encoding = np.eye(len(charset))

  def forward(self, inp):
    out = None #np.zeros(len(inp_str), len(c))
    for char in inp:
      i = self.keys.index(char)
      if out is None:
        out = self.encoding[i]
      else:
        out = np.concatenate((out, self.encoding[i]))
    return out #np.reshape(out, (1,len(out)))


class Linear(Layer):
  def __init__(self, inp_size, d):
    # for every hidden neuron
    self.weights = np.random.uniform(-0.1, 0.1, (d, inp_size))
    self.bias = np.random.uniform(-0.1, 0.1, d)
    self.inp = None
    self.d_o = None
    self.d = d
    self.d_weights = np.zeros((d, inp_size))
    self.d_bias = np.zeros(d)

  def forward(self, inp):
    # 1 x 507 * 100 * 507 (need transpose of weights) = 1 x 100
    self.inp = inp
    # np.matmul automatically adds dimensions we need for multiplication
    return np.matmul(self.weights, inp) + self.bias 

  def backward(self, d_o):
    self.d_o = d_o
    return np.matmul(self.weights.T, d_o);

  def update(self, eta):
    d_o = self.d_o.reshape((-1, 1))
    inp = self.inp.reshape((-1, 1))
    self.d_weights = np.matmul(d_o, inp.T)# + self.d_weights # momentum
    self.d_bias = self.d_o# self.d_bias * 0.9
    self.weights = (self.weights - self.d_weights * eta)# * (1.0-0.00005) # weight decay
    self.bias = (self.bias - self.d_bias * eta)# * (1.0-0.00005)

  def test(self):
    print('---Linear Test---')
    inp = np.array([1.0, 0.5]) 
    self.weights = np.array([[0.3, 0.2], [0.5, 0.3], [0.0, 0.0]]) 
    self.bias = np.array([0.1, 0.4, 0.1]) 
    out = self.forward(inp)
    target = np.array([1.0, 0.0, 0.0]) 
    diff = np.array([-1.0, -10.0, 0.0])
    back = self.backward(diff)
    u = self.update(.1)
    return out

class Sigmoid(Layer):
  def __init__(self):
    # 1 / 1 + e ^ - z
    self.out = None

  def forward(self, inp):
    self.out = 1.0 / (1.0 + np.exp(-inp))
    return self.out

  def backward(self, d_o):
    return d_o * self.out * (1 - self.out)
  
  def test(self):
    print('---Sigmoid Test---')
    inp = np.array([500, 100, -100, -500]) 
    out = self.forward(inp)
    print(inp)
    print(out)
    return out


class Softmax(Layer):
  def __init__(self):
    # e ^ i / sum(e ^ i)
    self.out = None

  def forward(self, inp):
    denom = sum(np.exp(inp))
    self.out = np.exp(inp)/denom
    return self.out

  def backward(self, d_o):
    jacobian = np.zeros((len(self.out), len(self.out))) # d_or is 3x1, deriv w.r.t. to all inp is 3x3 
    for i in range(len(self.out)): # for each index in the error
      for j in range(len(self.out)): # for each index in the softmax output
        if i==j: # if they are the same index, this deriv should encourage the output
            jacobian[i][j] = self.out[i] - self.out[i] * self.out[i];
        else: # if different, they should be penalized
            jacobian[i][j] = -self.out[i] * self.out[j]

    d_i = np.matmul(jacobian, d_o)
    return d_i 

  def update(self, eta):
    # no weights
    pass

  def test(self):
    print('---Softmax Test---')
    inp = np.array([0.9, 0.5, 0.5]) 
    out = self.forward(inp)
    print(inp)
    print(out)
    return out


def square_error(inp, labels):
  diff = inp - labels 
  return sum(0.5 * (np.power(diff, 2))), diff


class Network(object):
  def __init__(self):
    self.layers = []

  def add(self, layer):
    self.layers.append(layer)

  def forward(self, inp):
    out = inp
    for layer in self.layers:
      out = layer.forward(out)
      #print(out.shape)
    #print(out)
    return out

  def backward(self, d_o):
    d_i = d_o 
    for layer in self.layers[::-1]:
      d_i = layer.backward(d_i)
      #print(d_i.shape)
    #print(d_i)
    return d_i 

  def update(self, eta):
    for layer in self.layers:
      layer.update(eta)

  def evaluate(self, data, label_encoder):
      tested = 0
      correct = 0
      for line in data:
        split = line.split(' ')
        lang = split[0] # Make label
        label = label_encoder.forward([lang])
        sentence = line[len(lang):] # Make sentence
        
        for start in range(len(sentence)-4):
          inp_str = sentence[start:start+5] 
          out = self.forward(inp_str)   
          pred_idx = np.argmax(out)
          if label[pred_idx] == 1.0:
            correct += 1
          tested += 1
      accuracy = correct / (1.0 * tested)
      return accuracy
