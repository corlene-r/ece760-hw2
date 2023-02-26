import numpy as np
from matplotlib import pyplot as plt, patches

# Tree creation: 
class Node:
   def __init__(self, split, left, right):
      self.left = left
      self.right = right
      self.split = split
class Leaf:
   def __init__(self, label):
      self.label = label

# Classifier Building
def FindCandidateSplits(feat, labels):
   # Give only 1 feature
   feats, labels = zip(*sorted(zip(feat, labels)))

   candidateSplits = []
   for i in range(len(labels) - 1):
      if labels[i] != labels[i + 1]: # If labels are different add candidate split
         candidateSplits.append(feats[i+1])
         if feats[i] == feats[i+1]: 
            # if label is same, append split at feature and closest unique val after
            for k in range(i+2, len(labels)):
               if feats[i] != feats[k]:
                  candidateSplits.append(feats[k])
                  break
   return np.unique(candidateSplits) # Only care about unique splits
def EntropyOfSplit(feat, labels, split, testingOn=False):
   labelsLow = labels[feat < split]
   labelsHigh = labels[feat >= split]

   log2 = lambda x: 0 if x == 0 else np.log2(x)

   nLo = len(labelsLow); 
   ones = np.sum(labelsLow);  zeros = nLo - ones
   if testingOn: print("\tnLo:", nLo, "\t\tones:", ones, "\tzeros:", zeros, "\tLLi:", labelsLow)
   LoE = -nLo/len(labels) * ((ones / nLo) * log2(ones / nLo) + (zeros / nLo) * log2(zeros / nLo)) \
           if nLo != 0 else 0

   nHi = len(labelsHigh); 
   ones = np.sum(labelsHigh);  zeros = nHi - ones
   if testingOn: print("\tnHi:", nHi, "\t\tones:", ones, "\tzeros:", zeros, "\tLhi:", labelsHigh)
   HiE = -nHi/len(labels) * ((ones / nHi) * log2(ones / nHi) + (zeros / nHi) * log2(zeros / nHi)) \
           if nHi != 0 else 0

   return LoE + HiE
def FindMaxEntropySplit(feat, labels):
   n = len(labels)
   ones = np.sum(labels);   zeros = n - ones
   if ones == 0 or zeros == 0: 
      return (-1, 0)
   E = - ones/n * np.log2(ones / n) - zeros/n * np.log2(zeros/n)

   splits    = FindCandidateSplits(feat, labels)
   splitEnts = np.array(list(map(lambda s: EntropyOfSplit(feat, labels, s), splits)))
   maxind    = np.argmax(E - splitEnts)

   if E - splitEnts[maxind] == 0: 
      return (-1, 0)
   else:
      return (splits[maxind], E - splitEnts[maxind])
def BuildTree(feats, labels): 
   (split0, gain0) = FindMaxEntropySplit(feats[:, 0], labels)
   (split1, gain1) = FindMaxEntropySplit(feats[:, 1], labels)

   # If no entropy gain, make a leaf
   if gain0 <= 0 and gain1 <= 0:
      ones = np.sum(labels)
      if (len(labels) - 1) / 2 >= ones: return Leaf(0)
      else:                       return Leaf(1)
   # If more gain splitting on 0th feature, split there; else split on 1st
   elif gain0 > gain1:
      labelsLo = labels[feats[:, 0] < split0];    labelsHi = labels[feats[:, 0] >= split0]
      featsLo  = feats[feats[:, 0] < split0];     featsHi  = feats[feats[:, 0] >= split0]
      return Node([0, split0], BuildTree(featsHi, labelsHi),  BuildTree(featsLo, labelsLo))
   else: 
      labelsLo = labels[feats[:, 1] < split1];    labelsHi = labels[feats[:, 1] >= split1]
      featsLo  = feats[feats[:, 1] < split1];     featsHi  = feats[feats[:, 1] >= split1]
      return Node([1, split1], BuildTree(featsHi, labelsHi),  BuildTree(featsLo, labelsLo))
def TraverseTree(feat, node):
   if type(node) == type(Leaf(-1)):
      return node.label
   else:
      return TraverseTree(feat, node.left if feat[node.split[0]] >= node.split[1] else node.right)
def printTree(root, markerStr="+- ", levelMarkers=[]):
# Reference: https://simonhessner.de/python-3-recursively-print-structured-tree-including-hierarchy-markers-using-depth-first-search/
   emptyStr = " "*len(markerStr)
   connectionStr = "|" + emptyStr[:-1]
   level = len(levelMarkers)
   mapper = lambda draw: connectionStr if draw else emptyStr
   markers = "".join(map(mapper, levelMarkers[:-1]))
   markers += markerStr if level > 0 else ""
   if type(root) == type(Leaf(-1)): 
      print(f"{markers}Label {root.label}")
   else:
      print(f"{markers}x{root.split[0]+1} = {root.split[1]}")
      printTree(root.right, markerStr, [*levelMarkers, True])
      printTree(root.left, markerStr, [*levelMarkers, False])
def PlotBoundary(root, xLo, yLo, xHi, yHi, ax): 
   # Needed to plot the boundary on a graph for the decision tree. 
   if type(root) == type(Leaf(-1)):
      if root.label == 0:
         ax.add_patch(patches.Rectangle((xLo,yLo), xHi - xLo, yHi - yLo, color=(1,0.7,0.7)))
      if root.label == 1:
         ax.add_patch(patches.Rectangle((xLo,yLo), xHi - xLo, yHi - yLo, color=(0.7,0.7,1)))
   else:
      if root.split[0] == 0:
         PlotBoundary(root.left,  root.split[1], yLo, xHi,           yHi, ax)
         PlotBoundary(root.right, xLo,           yLo, root.split[1], yHi, ax)
      elif root.split[0] == 1:
         PlotBoundary(root.left,  xLo, root.split[1], xHi, yHi,           ax)
         PlotBoundary(root.right, xLo, yLo,           xHi, root.split[1], ax)
def CountNodes(root):
   if type(root) == type(Leaf(-1)):
      return 1
   else:
      return 1 + CountNodes(root.left) + CountNodes(root.right)
      
# Testing Functions:
def TestFindSplits():
   TestData   = np.array([0,1,2,3,4,5,6,7,8,9])
   TestLabels = np.array([0,0,0,1,1,1,0,0,1,0])

   print("Found:\t", FindCandidateSplits(TestData, TestLabels), "\tShould be:\t", "[3, 6, 8, 9]")


   TestData   = np.array([0,1,2,2,3,3,3,4,5,5])
   TestLabels = np.array([0,0,0,1,1,1,0,0,1,0])

   print("Found:\t", FindCandidateSplits(TestData, TestLabels), "\tShould be:\t", "[2, 3, 4, 5]")

   TestData   = np.array([0,1,2,2,3,3,3,4,5,5])
   TestLabels = np.array([0,0,0,0,0,0,0,0,0,0])

   print("Found:\t", FindCandidateSplits(TestData, TestLabels), "\t\tShould be:\t", "[]")
def TestEntropyOfSplit():
   D = np.array([0,1,2,3,4,5])
   L = np.array([0,0,0,1,1,1])

   print("E Split @ 1: ", EntropyOfSplit(D, L, 1, testingOn=True))
   print("0 E Split:   ", EntropyOfSplit(D, L, 3, testingOn=True))
   print("1 E Split:   ", EntropyOfSplit(D, L, 0, testingOn=True))
def TestFindMaxEntropySplit():
   TestData   = np.array([0,1,2,3,4,5,6,7,8,9])
   TestLabels = np.array([0,0,0,1,1,1,0,0,1,0])

   print("Found:\t", FindMaxEntropySplit(TestData, TestLabels))


   TestData   = np.array([0,1,2,2,3,3,3,4,5,5])
   TestLabels = np.array([0,0,0,1,1,1,0,0,1,0])

   print("Found:\t", FindMaxEntropySplit(TestData, TestLabels))

   TestData   = np.array([0,1,2,2,3,3,3,4,5,5])
   TestLabels = np.array([0,0,0,0,0,0,0,0,0,0])

   print("Found:\t", FindMaxEntropySplit(TestData, TestLabels))

   TestData   = np.array([0,1,2,3,4,5,6,7,8,9])
   TestLabels = np.array([0,0,0,0,0,1,1,1,1,1])

   print("Found:\t", FindMaxEntropySplit(TestData, TestLabels))
def TestBuildTraverseTree():
   TestData   = np.array([[0,1,2,3,4,5,6,7,8,9], [0,0,0,0,0,0,0,0,0,0]]).T
   TestLabels = np.array( [0,0,0,1,1,1,0,0,1,0])

   Tree = BuildTree(TestData, TestLabels)
   print("Classifying\n\t1:", TraverseTree([1,0], Tree), "\n\t3:", TraverseTree([3,0], Tree), \
          "\n\t6:", TraverseTree([6,0], Tree), "\n\t8:", TraverseTree([8,0], Tree), 
          "\n\t9:", TraverseTree([9,0], Tree), "\n\t\t(Should be 0, 1, 0, 1, 0)")
   printTree(Tree)

   TestData   = np.array([[0,1,2,2,3,3,3,4,5,5], [0,1,2,3,4,5,6,7,8,9]]).T
   TestLabels = np.array( [0,0,0,0,0,0,0,0,0,0])

   print("Tree that has all 0's:")
   printTree(BuildTree(TestData, TestLabels))

   TestData   = np.array([[1,1,1], [5,5,5]]).T
   TestLabels = np.array( [1,0,0])
   print("Tree that should have 0 splits (with 1 '1' and 2 '0'):")
   printTree(BuildTree(TestData, TestLabels))

   TestData   = np.array([[1,1,1,1], [5,5,5,5]]).T
   TestLabels = np.array( [1,0,0,1])
   print("Tree that should have 0 splits (with 2 '1' and 2 '0'):")
   printTree(BuildTree(TestData, TestLabels))

   TestData   = np.array([[1,1,1], [5,5,5]]).T
   TestLabels = np.array( [1,0,1])
   print("Tree that should have 0 splits (with 2 '1' and 1 '0'):")
   printTree(BuildTree(TestData, TestLabels))

# Problemset
def P2():
   print("\n----------------------------Problem 2----------------------------")
   data = np.array([[1,1],[1,1]])
   labels = np.array([1, 0])
   Tree = BuildTree(data,labels)
   print("Testing for Data = [[1,1],[1,1]], Labels = [1, 0]\n")
   printTree(Tree)
def P3(): 
   print("\n----------------------------Problem 3----------------------------")
   Data = np.loadtxt("Druns.txt", dtype=float)
   feats = Data[:, :2]
   labels = Data[:, 2].astype(int)

   split0s    = FindCandidateSplits(feats[:, 0], labels)
   split1s    = FindCandidateSplits(feats[:, 1], labels)
   split0Ents = np.array(list(map(lambda s: EntropyOfSplit(feats[:, 0], labels, s), split0s)))
   split1Ents = np.array(list(map(lambda s: EntropyOfSplit(feats[:, 1], labels, s), split1s)))

   n = len(labels)
   ones = np.sum(labels);   zeros = n - ones
   E = - ones/n * np.log2(ones / n) - zeros/n * np.log2(zeros/n)
   split0GainRs = [(E - e) / e for e in split0Ents]
   split1GainRs = [(E - e) / e for e in split1Ents]

   print("\nSplit Value (top) with entropy (bottom) along x_1:\n", np.vstack((split0s, split0GainRs)),\
         "\nSplit Value (top) with entropy (bottom) along x_2:\n", np.vstack((split1s, split1GainRs)), "\n")
def P4(): 
   print("\n----------------------------Problem 4----------------------------")
   Data = np.loadtxt("D3leaves.txt", dtype=float)
   feats = Data[:, :2]
   labels = Data[:, 2].astype(int)

   Tree = BuildTree(feats, labels)
   print("\nTree for D3Leaves.txt:\n")
   printTree(Tree)
   print("\n\n")
def P5(): 
   print("\n----------------------------Problem 5----------------------------")
   Data = np.loadtxt("D1.txt", dtype=float)
   feats = Data[:, :2]
   labels = Data[:, 2].astype(int)

   Tree = BuildTree(feats, labels)
   print("Tree for D1.txt:")
   printTree(Tree)

   Data = np.loadtxt("D2.txt", dtype=float)
   feats = Data[:, :2]
   labels = Data[:, 2].astype(int)

   Tree = BuildTree(feats, labels)
   print("\n\nTree for D2.txt:")
   printTree(Tree)
def P6():
   print("\n----------------------------Problem 6----------------------------")
   print("This problem saves images; look for those with the prefix 'P1.6'.")
   Data = np.loadtxt("D1.txt", dtype=float)
   feats = Data[:, :2]
   labels = Data[:, 2].astype(int)

   label0s = feats[labels == 0, :]
   label1s = feats[labels == 1, :]

   plt.scatter(label0s[:,0],label0s[:,1],color='red',label='Label 0')
   plt.scatter(label1s[:,0],label1s[:,1],color='blue',label='Label 1')
   plt.title("Scatterplot of Data for D1.txt")
   plt.legend()
   plt.savefig("P1.6D1.png")

   fig, ax = plt.subplots()

   Tree = BuildTree(feats, labels)
   PlotBoundary(Tree, np.min(Data[:, 0]) - 1, np.min(Data[:, 1]) - 1, \
                      np.max(Data[:, 0]) + 1, np.max(Data[:, 1]) + 1, ax)
   plt.title("Decision Tree Boundary for D1.txt")
   plt.savefig("P1.6D1Bounds.png")

   Data = np.loadtxt("D2.txt", dtype=float)
   feats = Data[:, :2]
   labels = Data[:, 2].astype(int)
   feats = Data[:, :2]
   labels = Data[:, 2].astype(int)

   label0s = feats[labels == 0, :]
   label1s = feats[labels == 1, :]

   plt.clf()
   plt.scatter(label0s[:,0],label0s[:,1],color='red',label='Label 0')
   plt.scatter(label1s[:,0],label1s[:,1],color='blue',label='Label 1')
   plt.title("Scatterplot of Data for D2.txt")
   plt.legend()
   plt.savefig("P1.6D2.png")

   fig, ax = plt.subplots()

   Tree = BuildTree(feats, labels)
   PlotBoundary(Tree, np.min(Data[:, 0]) - 1, np.min(Data[:, 1]) - 1, \
                      np.max(Data[:, 0]) + 1, np.max(Data[:, 1]) + 1, ax)
   plt.title("Decision Tree Boundary for D2.txt")
   plt.savefig("P1.6D2Bounds.png")
def P7():
   print("\n----------------------------Problem 7----------------------------")
   print("This problem saves images; look for those with the prefix 'P1.7'.")
   Data = np.loadtxt("Dbig.txt", dtype=float)
   feats = Data[:, :2]
   labels = Data[:, 2].astype(int)
   (xLo, yLo, xHi, yHi) = (np.min(feats[:,0]), np.min(feats[:,1]), \
                           np.max(feats[:,0]), np.max(feats[:,1]))

   idxs = np.random.permutation(10000)

   nnodes = []
   errs = []
   for trn in [32, 128, 512, 2048, 8192]:
      Tree = BuildTree(feats[idxs[:trn], :], labels[idxs[:trn]])
      pred = [TraverseTree(f, Tree) for f in feats[idxs[trn:], :]]
      error_vec = [0 if i[0]==i[1] else 1 for i in np.vstack((pred, labels[idxs[trn:]])).T]
      print("For D" + str(trn) + ":")
      print("\tNumber of Errors:\t", np.sum(error_vec))
      print("\tProb of Error:\t\t", 100*np.sum(error_vec)/(10000 - trn))
      print("\tNumber of Nodes:\t", CountNodes(Tree), "\n")
      nnodes.append(trn)
      errs.append(100*np.sum(error_vec)/(10000 - trn))
      fig, ax = plt.subplots()
      plt.xlim([xLo, xHi])
      plt.ylim([yLo, yHi])
      PlotBoundary(Tree, xLo, yLo, xHi, yHi, ax)
      plt.title("Boundary Region for D" + str(trn))
      plt.savefig("P1.7D" + str(trn) + ".png")

   plt.clf()
   plt.plot(nnodes, errs)
   plt.title("Training Curve for Decision Tree")
   plt.xlabel("Number of Training Nodes")
   plt.ylabel("Test Set Error (percentage)")
   plt.savefig("P1.7.png")


P2()
P3()
P4()
P5()
P6()
P7()