import os
import random
import math
import argparse
import pickle

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
(where  Line i   realphabets  the  content  of  the  i  th  line  in  the  test  file,  and  Language i   is  the
language  determined  as  most  likely).  Submit  this  file  with  your  assignment.
"""

d = 100
eta = 0.001

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
  enc = 'latin8'# if 'test' in file_name else 'utf-8'
  #line.decode('iso-8859-1')
  with open(file_name, 'r', encoding=enc) as f:
    for line in f:
      lines.append(line)
    #lines = [line[:-1].split(' ') for line in lines]

  return lines

def increment_dict(inp_dict, token):
  if token not in inp_dict:
    inp_dict[token] = 1 
  else:
    inp_dict[token] += 1
  return inp_dict 

def count(files):
  languages = {}
  alphabet = {}
  for i, f in enumerate(files):
    for line in f:
      for token in line:
        alphabet = increment_dict(alphabet, token)
      if i < 2:
        lang = line.split(' ')[0]
        languages = increment_dict(languages, lang)
  return alphabet, languages


# MAIN
if __name__ == "__main__":
  args = parse_arg() # Parse paths to data files

  train = load(args.train_file) # Load these files from disk
  dev = load(args.dev_file)
  test = load(args.test_file)

  alphabet, langs = count([train, dev, test]) # Count alphabet and languages
  print("--- Characters Found ---")
  print(alphabet)
  print("--- Languages Found ---")
  print(langs)

  if os.path.exists('model.p'):
    model = pickle.load(open("model.p", "rb"))
    label_encoder = pickle.load(open("label_encoder.p", "rb"))
  else:
    label_encoder = Encoder(1, langs) # Make language encoder

    model = Network() # Make the neural network 
    model.add(Encoder(5, alphabet))
    model.add(Linear(5*len(alphabet), d))
    model.add(Sigmoid())
    model.add(Linear(d, 3))
    model.add(Softmax())

  dev_accuracy = []
  train_accuracy = []
  epochs = 3
  for epoch in range(epochs):
    # Test on dev and train data
    #dev_accuracy.append(model.evaluate(train, label_encoder))
    #print("Dev Accuracy: %4f" % (dev_accuracy[-1]))
    #train_accuracy.append(model.evaluate(train, label_encoder))
    #print("Train Accuracy: %4f" % (train_accuracy[-1]))

    size = len(train)
    correct = 0
    tested = 0
    for i in range(size):
      line = random.choice(train) # SGD
      split = line.split(' ')
      lang = split[0] # Make label
      label = label_encoder.forward([lang])
      sentence = line[len(lang):] # Make sentence
      sections = [sentence[start:start+5] for start in range(len(sentence)-4)] # Carve into sections
      random.shuffle(sections)
      
      for inp_str in sections:
        out = model.forward(inp_str)   
        loss, d_o = square_error(out, label)
        model.backward(d_o)
        model.update(eta)

        pred_idx = np.argmax(out)
        if label[pred_idx] == 1.0:
          correct += 1
        tested += 1
        accuracy = correct / (1.0 * tested)
      
      if i % 100 == 0:
        print("%4d / %4d Sentences. Accuracy: %4f" % (i, size, accuracy))

    pickle.dump(model, open("model.p", "wb")) # Save model for later
    pickle.dump(dev_accuracy, open("da.p", "wb")) # Save model for later
    pickle.dump(train_accuracy, open("ta.p", "wb")) # Save model for later
    pickle.dump(label_encoder, open("label_encoder.p", "wb")) # Save model for later





