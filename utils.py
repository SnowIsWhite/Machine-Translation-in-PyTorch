import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m,s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points, fname):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig(fname + '_loss.png')

def beamSearch(topv, topi, prev_array, prev_prob):
    temp_array = prev_array
    temp_array.append(topi[0][0])
    arr1 = temp_array
    temp_array = prev_array
    temp_array.append(topi[0][1])
    arr2 = temp_array

    temp_prob = prev_prob
    prob1 = temp_prob + topv[0][0]
    temp_prob = prev_prob
    prob2 = temp_prob + topv[0][1]
    print(arr1)
    print(arr2)
    return [(arr1, prob1), (arr2, prob2)]

def saveTranslatedResults(results, fname):
    with open(fname, 'w') as f:
        for tup in results:
            f.write(' '.join(tup[0]) + '\n')
            f.write(' '.join(tup[1]) + '\n\n')
