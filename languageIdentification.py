import os
import argparse
import math
from sklearn.preprocessing import OneHotEncoder
import numpy as np

from net import *

# Name: Richard Higgins
# Uniqname: Relh

"""
Before you begin training the neural net, calculate the accuracy of the classifier on the
training data and the accuracy of the classifier on the dev data. While you are training the
neural net, at the end of each epoch, calculate these accuracies again. Finally, produce a
graph of training epoch v. accuracy (plot training accuracy and dev accuracy on the same
graph). Save  this  graph  as  accuracy.png,  and  submit  this  file  with  your  assignment.
4. Open the test file, provided as the third argument on the command line, and for each line in
the  test  file,  use  the  neural  network  to  identify  the  probable  language  for  that  sentence.
5. Produce  a   file  called  languageIdentificationPart1.output,  with  the  following  content:
Line1  Language1
Line2  Language2
...
LineN  LanguageN
(where  Line i   represents  the  content  of  the  i  th  line  in  the  test  file,  and  Language i   is  the
language  determined  as  most  likely).  Submit  this  file  with  your  assignment.
"""

d = 100
eta = 0.001

def mse(y, y_hat):
  return 0.5 * ((y - y_hat) ** 2)


def parse_arg(): # parses all the command line arguments
    parser = argparse.ArgumentParser('SBD')
    parser.add_argument('--train_file', type=str, default='languageIdentification.data/train', help='training file')
    parser.add_argument('--dev_file', type=str, default='languageIdentification.data/dev', help='dev file')
    parser.add_argument('--test_file', type=str, default='languageIdentification.data/test', help='test file')

    args = parser.parse_args()
    if not os.path.exists(args.train_file) or not os.path.exists(args.test_file):
        parser.error('Either your training or test file does not exist'.format())
    return args

    print >>f, ' '.join(line)


def load(file_name): # loads a .train .dev .test input file
  lines = []
  enc = 'latin8' if 'test' in file_name else 'utf-8'
  with open(file_name, 'r', encoding=enc) as f:
    for line in f:
      lines.append(line)
    #lines = [line[:-1].split(' ') for line in lines]

  return lines


def is_char_n_eos(line, char):
    return (line[1][-1] == char and \
       (line[2] == 'EOS' or line[2] == 'NEOS')) # counts periods that are the last element in the token


def count_c(files):
  present = {}
  for f in files:
    for line in f:
      for char in line:
        if char not in present:
          present[char] = 1 
        else:
          present[char] += 1
  return present


# MAIN
if __name__ == "__main__":
  args = parse_arg()

  train = load(args.train_file)
  dev = load(args.dev_file)
  test = load(args.test_file)

  c = count_c([train, dev, test])
  print("A dict of the characters found")
  print(c)

  inp_str = 'abcde'
  #inp_enc = encode_input(inp_str)

  # inp_enc is now the input!
  model = Network()
  model.add(Encoder(5, c))
  model.add(Linear(5*len(c), d))

  model.forward(inp_str)   





