import numpy as np
import pandas as pd

class Node():

    # My node contains the feature it splits upon as well as either the 
    # possible categories in the case of a categorical variable, or the threshold 
    # to dichotomize on
    def __init__(self, feature, iscat, tc, c, sse = None) -> None:
        self.feature = feature

        self.iscat = iscat

        #if not categorical left (0) is less than
        self.children = c

        # if categorical build a dictionary for the values
        self.categories = tc if iscat else None
        # If not categorical dichotomize
        self.threshold = None if iscat else tc

        self.sse = sse

    # This function takes in a dataframe and then applies the filter that corresponds
    # to the given branch
    def apply_filter(self, df, cid):
        if self.iscat:
            catname = next(key for key, value in self.categories.items() if value == cid)
            retdf = df[df[self.feature] == catname]

        else:
            if cid == 0:
                retdf = df[df[self.feature] < self.threshold]
            else:
                retdf = df[df[self.feature] >= self.threshold]

        return retdf


class DecisionTree():

    def __init__(self) -> None:
        self.target = None
        self.preds = None

        self.root = None
        self.trainds = None

        self.isfit = False

    def add_node(self, parent, cid, child):
        if parent == None:
            self.root = child
            return self.root

        else:
            parent.children[cid] = child
            return parent.children[cid]

    
    def test_node(self, parent, cid, child):
        hold = None
        if parent == None:
            self.root = child
        
        else:
            hold = parent.children[cid]
            parent.children[cid] = child

        sse = self.eval_tree()

        if hold == None:
            self.root = None
        else:
            parent.children[cid] = hold

        return sse


    def predict(self, df):
        res = []

        for r in df.iterrows():
            r = r[1]
            n = self.root

            while type(n) == Node:
                if n.iscat:
                    n = n.children[n.categories[r[n.feature]]]

                else:
                    cond = r[n.feature] < n.threshold
                    if cond:
                        n = n.children[0]
                    else:
                        n = n.children[1]

            res.append(n)

        res = np.array(res)

        return res

    def eval_tree(self):
        res = self.predict(self.trainds)

        SSE = np.sum((res - self.trainds[self.target])**2)

        return SSE


    def fit(self, ds, target, preds, n2fit = 25):
        # Save everything
        self.target = target
        self.preds = preds
        self.trainds = ds

        # Build up a node queue and an accompanying dataset queue so we can
        # build this tree breadth first
        nq = [None]
        dsq = [ds]

        # Currently I'm just fitting a set number of nodes
        for i in range(n2fit):

            # Pull out the current working data and node
            currparent = nq[0]
            currds = dsq[0]

            # If this is the first one, we have to be careful because The tree will be None
            if currparent == None:
                jrange = 1
            else:
                jrange = len(currparent.children)

            # This will add a new node to every branch
            for currchild in range(jrange):
                # Going to test each type of node on each branch
                ns = []
                scores = []

                # Need to filter the dataset for the given branch
                subcurrds = currds
                if currparent != None:
                    subcurrds = currparent.apply_filter(currds, currchild)

                # Now for each predictive variable let's fit a new node
                for pvar in preds:
                    # Call it categorical if there are less than 10 unique values
                    if len(pd.unique(ds[pvar])) < 10:
                        cs, tc, sse = self.package_categorical(subcurrds, pvar)
                        n = Node(pvar, True, tc, cs, sse)
                    # Otherwise it's continuous
                    else:
                        thisthresh = np.mean(subcurrds[pvar])
                        cs, tc, sse = self.package_continuous(subcurrds, pvar, thisthresh)

                        n = Node(pvar, False, tc, cs, sse)
                    
                    ns.append(n)
                    scores.append(self.test_node(currparent, currchild, n))
                
                # We want to add the node which lowers the SSE the most we're greedy
                idtoadd = np.argmin(scores)
                n2add = ns[idtoadd]
                print('Adding Node ' + str(n2add.feature) + ' with SSE ' + str(scores[idtoadd]))

                # Add the node to the tree and add it to the explore queue
                self.add_node(currparent, currchild, n2add)
                nq.append(n2add)

                # At the root just give the next generation the same df
                if currparent == None:
                    dsq.append(currds)
                else:
                    dsq.append(currparent.apply_filter(currds, currchild))
            # Pop the finished node out
            nq = nq[1:]
            dsq = dsq[1:]

        self.fit = True


    def package_categorical(self, set, feature):
        names = pd.unique(set[feature])
        retd = {}
        retc = []
        sse = 0

        for (i, n) in enumerate(names):
            retd[n] = i
            subset = set.loc[set[feature] == n, self.target]
            retc.append(np.mean(subset))

            sse += np.sum((np.mean(subset)*np.ones((len(subset),)) - subset)**2)

        return retc, retd, sse
    
    def package_continuous(self, set, feature, thresh):
        c = [np.mean(set.loc[set[feature] < thresh, self.target]),\
             np.mean(set.loc[set[feature] >= thresh, self.target])]
        
        sse = (set.loc[set[feature] < thresh, self.target] - \
               np.mean(set.loc[set[feature] < thresh, self.target]))**2 + \
               (set.loc[set[feature] >= thresh, self.target] - \
                np.mean(set.loc[set[feature] >= thresh, self.target]))**2

        return c, thresh, sse

        





    