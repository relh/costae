import glob
import pickle

# Name: Richard Higgins
# Uniqname: Relh

for ta, da in zip(glob.glob('ta*.p'), glob.glob('da*.p')):
  print('-Accuracy-')
  t_acc = pickle.load(open(ta,'rb'))
  d_acc = pickle.load(open(da,'rb'))
  print(t_acc)
  print(d_acc)
