from sys import argv
from os import mkdir, path
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.font_manager import FontProperties

# Timing Decorator
def timeFunction(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print '%s Function Took: \t %0.3f s' % (f.func_name.title(), (time2-time1))
        return ret
    return wrap

class dataAnalysisClass:
    # General Init functions
    def __init__(self, fileName, size=0):
        self.inputFileName = fileName
        self.loadData(size)
        self.createOutputFolder()
        self.avg = np.average(self.data)
        self.var = np.var(self.data)
        self.std = np.std(self.data)

    def loadData(self, size=0):
        if size != 0:
            with open(self.inputFileName) as inputFile:
                self.data = np.zeros(size)
                for x in xrange(size):
                    self.data[x] = float(next(inputFile))
        else:
            self.data = np.loadtxt(self.inputFileName)

    # Statistical Analysis with Multiple Methods
    def runAllAnalyses(self):
        if len(self.data) <= 100000:
            print "Autocorrelation..."
            self.autocorrelation()
        print "Bootstrap..."
        self.bootstrap()
        print "Jackknife..."
        self.jackknife()
        print "Blocking..."
        self.blocking()

    # Standard Autocorrelation
    @timeFunction
    def autocorrelation(self):
        self.acf = np.zeros(len(self.data)/2)
        for k in range(0, len(self.data)/2):
            self.acf[k] = np.corrcoef(np.array([self.data[0:len(self.data)-k], \
                                            self.data[k:len(self.data)]]))[0,1]

    # Bootstrap
    @timeFunction
    def bootstrap(self, nBoots = 1000):
        bootVec = np.zeros(nBoots)
        for k in range(0,nBoots):
            bootVec[k] = np.average(np.random.choice(self.data, len(self.data)))
        self.bootAvg = np.average(bootVec)
        self.bootVar = np.var(bootVec)
        self.bootStd = np.std(bootVec)

    # Jackknife
    @timeFunction
    def jackknife(self):
        jackknVec = np.zeros(len(self.data))
        for k in range(0,len(self.data)):
            jackknVec[k] = np.average(np.delete(self.data, k))
        self.jackknAvg = self.avg - (len(self.data) - 1) * (np.average(jackknVec) - self.avg)
        self.jackknVar = float(len(self.data) - 1) * np.var(jackknVec)
        self.jackknStd = np.sqrt(self.jackknVar)

    # Blocking
    @timeFunction
    def blocking(self, blockSizeMax = 500):
        blockSizeMin = 1

        self.blockSizes = []
        self.meanVec = []
        self.varVec = []

        for i in range(blockSizeMin, blockSizeMax):
            if(len(self.data) % i != 0):
                pass#continue
            blockSize = i
            meanTempVec = []
            varTempVec = []
            startPoint = 0
            endPoint = blockSize

            while endPoint <= len(self.data):
                meanTempVec.append(np.average(self.data[startPoint:endPoint]))
                startPoint = endPoint
                endPoint += blockSize
            mean, var = np.average(meanTempVec), np.var(meanTempVec)/len(meanTempVec)
            self.meanVec.append(mean)
            self.varVec.append(var)
            self.blockSizes.append(blockSize)

        self.blockingAvg = np.average(self.meanVec[-200:])
        self.blockingVar = (np.average(self.varVec[-200:]))
        self.blockingStd = np.sqrt(self.blockingVar)



    # Plot of Data, Autocorrelation Function and Histogram
    def plotAll(self):
        self.createOutputFolder()
        if len(self.data) <= 100000:
            self.plotAutocorrelation()
        self.plotData()
        self.plotHistogram()
        self.plotBlocking()

    # Create Output Plots Folder
    def createOutputFolder(self):
        self.outName = self.inputFileName[:-4]
        if not path.exists(self.outName):
            mkdir(self.outName)

    # Plot the Dataset, Mean and Std
    def plotData(self):
        # Far away plot
        font = {'fontname':'serif'}
        plt.plot(range(0, len(self.data)), self.data, 'r-', linewidth=1)
        plt.plot([0, len(self.data)], [self.avg, self.avg], 'b-', linewidth=1)
        plt.plot([0, len(self.data)], [self.avg + self.std, self.avg + self.std], 'g--', linewidth=1)
        plt.plot([0, len(self.data)], [self.avg - self.std, self.avg - self.std], 'g--', linewidth=1)
        plt.ylim(self.avg - 5*self.std, self.avg + 5*self.std)
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        plt.xlim(0, len(self.data))
        plt.ylabel(self.outName.title() + ' Monte Carlo Evolution', **font)
        plt.xlabel('MonteCarlo History', **font)
        plt.title(self.outName.title(), **font)
        plt.savefig(self.outName + "/data.eps")
        plt.savefig(self.outName + "/data.png")
        plt.clf()

    # Plot Histogram of Dataset and Gaussian around it
    def plotHistogram(self):
        binNumber = 50
        font = {'fontname':'serif'}
        count, bins, ignore = plt.hist(self.data, bins=np.linspace(self.avg - 5*self.std, self.avg + 5*self.std, binNumber))
        plt.plot([self.avg, self.avg], [0,np.max(count)+10], 'b-', linewidth=1)
        plt.ylim(0,np.max(count)+10)
        plt.ylabel(self.outName.title() + ' Histogram', **font)
        plt.xlabel(self.outName.title() , **font)
        plt.title('Counts', **font)

        #gaussian
        norm = 0
        for i in range(0,len(bins)-1):
            norm += (bins[i+1]-bins[i])*count[i]
        plt.plot(bins,  norm/(self.std * np.sqrt(2 * np.pi)) * np.exp( - (bins - self.avg)**2 / (2 * self.std**2) ), linewidth=1, color='r')
        plt.savefig(self.outName + "/hist.eps")
        plt.savefig(self.outName + "/hist.png")
        plt.clf()

    # Plot the Autocorrelation Function
    def plotAutocorrelation(self):
        font = {'fontname':'serif'}
        plt.plot(range(1, len(self.data)/2), self.acf[1:], 'r-')
        plt.ylim(-1, 1)
        plt.xlim(0, len(self.data)/2)
        plt.ylabel('Autocorrelation Function', **font)
        plt.xlabel('Lag', **font)
        plt.title('Autocorrelation', **font)
        plt.savefig(self.outName + "/autocorrelation.eps")
        plt.savefig(self.outName + "/autocorrelation.png")
        plt.clf()

    def plotBlocking(self):
        font = {'fontname':'serif'}
        plt.plot(self.blockSizes, self.varVec, 'r-')
        plt.ylabel('Variance', **font)
        plt.xlabel('Block Size', **font)
        plt.title('Blocking', **font)
        plt.savefig(self.outName + "/blocking.eps")
        plt.savefig(self.outName + "/blocking.png")
        plt.clf()

    # Print Stuff to the Terminal
    def printOutput(self):
        print "\nSample Size:    \t", len(self.data)
        print "\n=========================================\n"
        print "Sample Average: \t", self.avg
        print "Sample Variance:\t", self.var
        print "Sample Std:     \t", self.std
        print "\n=========================================\n"
        print "Bootstrap Average: \t", self.bootAvg
        print "Bootstrap Variance:\t", self.bootVar
        print "Bootstrap Error:   \t", self.bootStd
        print "\n=========================================\n"
        print "Jackknife Average: \t", self.jackknAvg
        print "Jackknife Variance:\t", self.jackknVar
        print "Jackknife Error:   \t", self.jackknStd
        print "\n=========================================\n"
        print "Blocking Average: \t", self.blockingAvg
        print "Blocking Variance:\t", self.blockingVar
        print "Blocking Error:   \t", self.blockingStd, "\n"




# Initialize the class
if len(argv) > 2:
    dataAnalysis = dataAnalysisClass(argv[1], int(argv[2]))
else:
    dataAnalysis = dataAnalysisClass(argv[1])

# Run Analyses
dataAnalysis.runAllAnalyses()

# Plot the data
dataAnalysis.plotAll()

# Print Some Output
dataAnalysis.printOutput()
