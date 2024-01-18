import numpy as np
import pandas as pd

class DBSCAN():

    def __init__(self, eps, minpts) -> None:
        self.eps = eps
        self.minpts = minpts

        self.scalemeans = None
        self.scalestds = None

        self.workingdata = None
        self.scaledworkingdata = None
        self.clusters = None

        self.contmaps = None
        self.discmaps = None

        self.ranges = None

    def fit(self, ds):
        scaleds = self.prescale(ds)

        clabels = np.nan*np.ones((scaleds.shape[0],))

        clstr = 0

        for p in range(scaleds.shape[0]):

            if not pd.isna(clabels[p]):
                continue 

            pt = scaleds[p, :]

            neighbours = self.neighbours(scaleds, pt)

            if len(neighbours) < self.minpts:
                clabels[p] = -1
                continue

            clstr += 1

            clabels[p] = clstr

            while len(neighbours) > 0:
                thisid = neighbours[0]
                
                pt2check = scaleds[thisid, :]

                if clabels[thisid] == -1:
                    clabels[thisid] = clstr

                if not pd.isna(clabels[thisid]):
                    neighbours = neighbours[1:]
                    continue

                clabels[thisid] = clstr

                subneighbours = self.neighbours(scaleds, pt2check)

                if len(subneighbours) > self.minpts:
                    [neighbours.append(s) for s in subneighbours if s not in neighbours]

                neighbours = neighbours[1:]

        self.workingdata = ds
        self.scaledworkingdata = scaleds
        self.clusters = clabels       

    def predict(self, ds):
        ds = self.usescale(ds)

        cs = []

        for i in range(ds.shape[0]):
            pt = ds[i, :]

            n = self.get_closest(self.scaledworkingdata, pt)

            cs.append(self.clusters[n])

        return cs


    def usescale(self, ds):
        retds = np.zeros((len(ds), len(ds.columns)))
        
        for i, col in enumerate(ds.columns):
            if np.issubdtype(ds[col], np.number):
                def mapfun(x):
                    return (x - self.contmaps[i][0])/self.contmaps[i][1]
                
                retds[:, i] = np.array(list(map(mapfun, ds[col])))
            else:
                def mapfun(x):
                    return self.discmaps[i][x]
                
                retds[:, i] = np.array(list(map(mapfun, ds[col])))

        return retds

    def prescale(self, ds):
        matrixds = np.zeros((len(ds), len(ds.columns)))

        contmaps = []
        discmaps = []

        for i, col in enumerate(ds.columns):
            if np.issubdtype(ds[col], np.number):
                matrixds[:, i] = (ds[col] - np.mean(ds[col]))/np.std(ds[col])

                contmaps.append([np.mean(ds[col]), np.std(ds[col])])
                discmaps.append(None)
                            
            else:
                mapdict = {}
                vs = pd.unique(ds[col])
                for (i, v) in enumerate(vs):
                    mapdict[v] = i

                def dictfun(x):
                    return mapdict[x]
                
                matrixds[:, i] = list(map(dictfun, ds[col]))
                discmaps.append(mapdict)
                contmaps.append(None)

        self.discmaps = discmaps
        self.contmaps = contmaps

        return matrixds
    
    def register_ranges(self, ds, targetname):
        self.ranges = {}
        for c in range(int(np.max(self.clusters))):
            cname = c + 1
            subds = ds[self.clusters == cname]

            self.ranges[cname] = [np.min(subds[targetname]), np.max(subds[targetname])]


    def get_closest(self, scaleds, p):
        # find the euclidean distance
        ds = np.linalg.norm(scaleds - p, axis=1)

        idtocheck = np.argmin(ds)

        while self.clusters[idtocheck] < 0 :
            ds[idtocheck] = np.inf
            idtocheck = np.argmin(ds)

        return np.argmin(ds)

    def neighbours(self, scaleds, p):
        # find the euclidean distance
        ds = np.linalg.norm(scaleds - p, axis=1)

        idx = np.where(ds < self.eps)[0]

        return list(idx)
