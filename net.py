import numpy as np


class Layer(object):
  def __init__(self, params):
    pass

  def forward(self, inp):
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
    self.weights = np.random.randn(d, inp_size)

  def forward(self, inp):
    # 1 x 507 * 100 * 507 (need transpose of weights) = 1 x 100
    # np.matmul automatically adds dimensions we need for multiplication
    return np.matmul(inp, self.weights.T)

class Network(object):
  def __init__(self):
    self.layers = []

  def add(self, layer):
    self.layers.append(layer)

  def forward(self, inp):
    out = inp
    for layer in self.layers:
      out = layer.forward(out)
      print(out.shape)
      #print(out)

  def backwards(self):
    pass

  def update(self):
    pass


