import pickle
import os

from languageIdentification import *

d = 100
eta = 0.1
model = pickle.load(open("model_{}_{}.p".format(d,eta), "rb"))
dev_acc = pickle.load(open("da_{}_{}.p".format(d,eta), "rb")) # Save model for later
train_acc = pickle.load(open("ta_{}_{}.p".format(d,eta), "rb")) # Save model for later
label_encoder = pickle.load(open("label_encoder_{}_{}.p".format(d,eta), "rb")) # Save model for later
data = load('languageIdentification.data/test')
answers = load('languageIdentification.data/test_solutions')
print(train_acc)
print(dev_acc)
#print(answers)

with open('languageIdentificationPart1.output', 'w') as f:
  tested = 0
  correct = 0
  for sentence, answer in zip(data, answers):
    preds = [0,0,0]
    for start in range(len(sentence)-5):
      inp_str = sentence[start:start+5] 
      out = model.forward(inp_str)   
      pred_idx = np.argmax(out)
      preds[pred_idx] += 1
    guess_idx = np.argmax(preds)
    if label_encoder.keys[guess_idx] == answer.split(' ')[-1].upper().strip():
      correct += 1
    tested += 1
    f.write(sentence[:-1] + " " + label_encoder.keys[guess_idx] + '\n')
  accuracy = correct / (1.0 * tested)
  print(accuracy)
