import numpy as np
from matplotlib import pyplot as plt, patches
from sklearn.tree import DecisionTreeClassifier

Data = np.loadtxt("Dbig.txt", dtype=float)
feats = Data[:, :2]
labels = Data[:, 2].astype(int)
(xLo, yLo, xHi, yHi) = (np.min(feats[:,0]), np.min(feats[:,1]), \
                        np.max(feats[:,0]), np.max(feats[:,1]))

idxs = np.random.permutation(10000)

nnodes = []
errs = []
for trn in [32, 128, 512, 2048, 8192]:
   clf = DecisionTreeClassifier()
   clf.fit(feats[idxs[:trn], :], labels[idxs[:trn]])

   err = 1 - clf.score(feats[idxs[trn:], :], labels[idxs[trn:]])
   # pred = [TraverseTree(f, Tree) for f in feats[idxs[trn:], :]]
   # error_vec = [0 if i[0]==i[1] else 1 for i in np.vstack((pred, labels[idxs[trn:]])).T]
   print("For D" + str(trn) + ":")
   print("\tPercent Errors:\t\t", 100*err)
   print("\tNumber of Nodes:\t", clf.tree_.node_count, "\n")
   nnodes.append(trn)
   errs.append(100*err)
   fig, ax = plt.subplots()

plt.clf()
plt.plot(nnodes, errs)
plt.title("Training Curve for Decision Tree")
plt.xlabel("Number of Training Nodes")
plt.ylabel("Test Set Error (percentage)")
plt.savefig("P2.png")
