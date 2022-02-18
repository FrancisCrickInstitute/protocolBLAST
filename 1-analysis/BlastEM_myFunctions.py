# Functions for BLAST EM analysis
import numpy as np
import pandas as pd
from copy import deepcopy
import math
import matplotlib.pyplot as plt

# %%
def sampleIDcheck(table):
    '''Make sure each sample appears only once'''
    sampleN = table.index.size
    uniqSampleN = table.index.nunique()
    if sampleN == uniqSampleN:
        print("All sample IDs are unique")
    else:
        print(f"! There are {sampleN} samples, but only {uniqSampleN} are unique!")


# %%
def scoreCheck(table, colArteIdx, colPerfIdx):
    '''perfect score check: if all artefacts scored as 0, then perfect should be scored as 1'''
    problemSample = []
    for i in np.arange(0, len(table.index)):
        if sum(table.iloc[i, colArteIdx]) == 0:
            trueScore = 1
        else:
            trueScore = 0
        
        if table.iloc[i, colPerfIdx] != trueScore:
            problemSample.append(table.index[i])
    
    if problemSample == []:
        print("'Perfect' scoring as expected, all good.")
    else:
        print(f"!Double check if {problemSample} is perfect or not!")


# %% 
def scoreSummary(table):
    '''Count how many samples are scored a certain value for each artefact type
    columns will be score counts'''
    summary = np.zeros((len(table.columns), 9))
    for arteIdx, arte in enumerate(table.columns):
        c = table[arte].value_counts(ascending=True)
        
        for j in c.index:
            summary[arteIdx, j] = c[j]
        
        summary[arteIdx, 5] = sum(summary[arteIdx, 0:5]) # count all entries, should = # of samples
        summary[arteIdx, 6] = sum(summary[arteIdx, 1:5]) # count all non-0
        summary[arteIdx, 7] = round(100*summary[arteIdx, 6]/summary[arteIdx, 5], 1) # % of samples with non-0 scoring
        summary[arteIdx, 8] = sum(summary[arteIdx, 1:4]) # count all uncertain scores ie not 0 (definitely not), not 4 (most certain)

    summary = pd.DataFrame(summary)
    summary.columns = ['0', '1', '2', '3', '4', 'total', 'non-0', 'non-0 %', 'uncertain (1,2,3)']
    summary.set_index(table.columns, inplace = True)

    # set 'uncertain' to 0 for 'perfect' because 0 = false, 1 = true, which differs from artefact scoring system where 1 means unlikely
    summary.loc['perfect', 'uncertain (1,2,3)'] = 0

    return summary

# %% 
def arteCooccurFreq(table, arteColsIdx, thr=None):
    '''How many samples have co-occuring artefacts'''
    # set default
    if thr == None:
        thr = 0
        print('Using default threshold = 0')
    
    arteCols = table.columns[arteColsIdx]
    tableTF = table[arteCols].gt(thr) # replace all >0 avlues with True, 0 with False
    arteCount = tableTF.sum(axis = 1) # count True across each row
    arteCountTable = pd.DataFrame(arteCount.value_counts().sort_index())
    arteCountTable.index.name = 'co-occur arte #'
    arteCountTable.columns = ['sample count']
    arteCountTable['% of total samples'] = round(arteCountTable['sample count']/len(table.index)*100, 1)

    v = arteCountTable.iloc[0:5, :].sum(axis = 0).to_list()
    arteCountTable.loc['total'] = [round(i) for i in v]
    
    return arteCountTable
    
# %%
def expected_duo_occurance_rate(scores, thr=None):
    '''Calculate the expected rate of 2 artefacts co-occuring together from occurance rate of artefacts, asumming independent'''
    if thr == None:
        thr = 0
        print('Using default thr = 0')
    
    # select columns
    colNames = np.arange(thr+1, int(max(scores.columns))+1)
    colNames = [str(i) for i in colNames]
    trueCount = scores[colNames].sum(axis = 1)
    
    # total number of samples & artefact occurance rate
    sampleN = scores.sum(axis = 1).unique()[0]
    arteRate = trueCount/sampleN
    
    # calculate px * py
    expectedDuo = np.zeros([len(arteRate.index), len(arteRate.index)])
    for idxi, i in enumerate(arteRate.index):
        for idxj, j in enumerate(arteRate.index):
            if idxi == idxj:
                pass
            else:
                expectedDuo[idxi, idxj] = round(arteRate[i] * arteRate[j] * sampleN, 0)
    
    return expectedDuo

# %%
def makeTable(sampleInfo, batchInfo, sampleID, sampleFeatures=None, batchFeatures=None):
    '''Create new table containing selected features of samples'''
    # default sample feature is all features
    if sampleFeatures == None:
        sampleFeatures = sampleInfo.columns
        print('Using all sample features by default.')
    
    # default batch feature is all features
    if batchFeatures == None:
        batchFeatures = batchInfo.columns
        print('Using all batch features by default.')

    # get sample info
    if 'group' not in sampleFeatures:
        sampleFeatures.append('group')
        print("Appending 'group' as a column")
        print(f'features sampled: {sampleFeatures}')
    
    sampleT = sampleInfo.loc[sampleID, sampleFeatures]
    
    # get batch info
    batchT = batchInfo.loc[sampleT['group']][batchFeatures]
    batchT.set_index(sampleT.index, inplace = True) # concat needs common index columns to work
    
    # join tables
    newTable = pd.concat([sampleT, batchT], axis = 1)
    
    return newTable


# %%
def getArteTF(table, arte, samples=None, thr=None):
    '''Create a True/False table of 3 columns: [specified-artefact, other-artefact, perfect] for samples'''
    # default sample is all
    if samples == None:
        samples = table.index
        print('Using all samples by default.')
    
    # default threshold is 0
    if thr == None:
        thr = 0
        print('Using thr = 0 by default.')
    
    scores = table.loc[samples, [arte, 'perfect']]
    # initialise empty table
    TF = pd.DataFrame(np.zeros((len(samples), 3)), index = samples, columns = [arte, 'other', 'perfect'])
    TF[arte][scores[arte].gt(thr)] = 1
    TF['other'][scores[arte].eq(0) & scores['perfect'].eq(0)] = 1
    TF['perfect'][scores['perfect'].eq(1)] = 1
    
    if sum(TF.sum(axis = 1)) != len(TF.index):
        print('!Row do not add up to 1, sth is wrong!')
    return TF

# %%
def allArteTF(scoreInfo, samples=None, thr=None):
    '''Create a True/False table of all artefacts for selected samples'''
    # default sample is all
    if samples == None:
        samples = scoreInfo.index
        print('Using all samples by default.')
    
    # default threshold is 0
    if thr == None:
        thr = 0
        print('Using thr = 0 by default.')
    
    # for selected samples & exlcuding scores for 'perfect'
    scores = scoreInfo.loc[samples, scoreInfo.columns != 'perfect']
    scoreTF = scores.gt(thr)
    return scoreTF


# %%
def TF_to_colNames(tableTF, replacingValues=None):
    '''Convert binary (1, 0) scoring table into single column of values, ONLY 1 true value allowed per row'''
    # use column names by default
    if replacingValues == None:
        replacingValues = tableTF.columns.to_list()
        print(f'Replacing with column names by default: {replacingValues}')
    
    # replace 1 with values
    t = tableTF
    for idx, cn in enumerate(tableTF.columns):
        t[cn].replace([1, 0], [replacingValues[idx], 'NaN'], inplace = True)
    
    # concatenate all columns into 1
    newTable = t.max(axis = 1)
    return newTable


# %% 
def scatter_3groups(scoreInfo, sampleInfo, batchInfo, arte, samples=None, sampleFeatures=None, batchFeatures=None, tf2type=None, type2num=None, thr=None, jFactor=None, jRandomSeed=None):
    '''Make a table that is ready to plot, jFactor=0 means no jitter'''
    # set defaults:
    if jFactor == None:
        jFactor = 0.05
        print(f'Using jFactor = {jFactor} by default')
    
    if jRandomSeed == None:
        jRandomSeed = 1
        print(f'Using jRandomSeed = {jRandomSeed} by default')
    
    # tf2type & type2num has to be supplied together
    if tf2type is not None and type2num == None:
        raise ValueError('if tf2num is supplied, type2num must be supplied too!')


    # isolate artefact scores, turn into TF by a threshold
    TF3col = getArteTF(scoreInfo, arte, samples, thr)
    TF1col = TF_to_colNames(TF3col, tf2type)

    # isolate feature data
    features = makeTable(sampleInfo, batchInfo, TF3col.index, sampleFeatures, batchFeatures)

    # concatenate into 1 table
    combined = pd.concat([TF1col, features], axis = 1)
    combined.rename(columns = {0: 'arte_type'}, inplace = True)

    # default conversion is:
    if type2num == None:
        type2num = {'perfect':1, 'other':2, arte:3}
        print(f'Using default tyep2num conversion: {type2num}')

    # replace str with numbers for plotting
    plotData = combined.copy(deep=True)
    for key, value in type2num.items():
        plotData['arte_type'].replace(key, value, inplace=True)

    # introduce y jitter, note jFactor = 0 means no jitter
    np.random.seed(jRandomSeed)
    j = np.random.randn(len(plotData.index))*jFactor
    plotData['arte_type_jitter'] = plotData['arte_type'] + j

    return combined, plotData


# %%
def count_arte_by_feature(scoreInfo, sampleInfo, batchInfo, featureName, arte=None, exclude=None, thr=None, inPercentage=None):
    '''Generate a count table: columns = artefact types, row = categories by specified feature
    If do not wish to discard samples, set discardSamples = []
    If want to have count rather than percentage, set inPercentage = False'''
    
    # set default
    if arte == None:
        arte = scoreInfo.columns.to_list()
        print('Including all artefacts by default')
    if exclude == None:
        exclude = sampleInfo.index[sampleInfo['discard'].eq(1)]
        print('Excluding discarded samples by default')
    if thr == None:
        thr = 0
        print('Using default True/ False threshold = 0 (T if >0, F if =< 0) ')
    if inPercentage == None:
        inPercentage = True
        print('Outputting % by default')
    
    # merge all into a big table
    allInfo = pd.concat([scoreInfo, makeTable(sampleInfo, batchInfo, scoreInfo.index)], axis = 1)
    allInfo.drop(exclude, inplace=True)
    
    # Count # of artefacts for each category
    cat = allInfo[featureName].unique()
    
    countTable = np.zeros((len(cat), len(arte)))
    totalCol = np.zeros((len(cat), 1))
    
    for catIdx, i in enumerate(cat):
        sampleIdx = allInfo[featureName].eq(i)
        totalCol[catIdx] = sum(sampleIdx == True)
        for arteIdx, j in enumerate(arte):
            countTable[catIdx, arteIdx] = sum(allInfo[j][sampleIdx].gt(thr))
    
    # calculate % samples that suffers from each artefact
    percentTable = np.round(countTable/totalCol*100, 1)
    
    colNames = deepcopy(arte)
    colNames.append('total n')

    if inPercentage == False:
        resultTable = pd.DataFrame(np.concatenate((countTable, totalCol), axis = 1), index = cat, columns = colNames)
    elif inPercentage == True:
        resultTable = pd.DataFrame(np.concatenate((percentTable, totalCol), axis = 1), index = cat, columns = colNames)


    
    return resultTable

# %%
def my_chain_list(list2chain):
    newList = []
    for i in np.arange(0, len(list2chain)):
        newList = newList + list2chain[i]
    return newList

# %%
def getSample(scoreInfo, sampleInfo, batchInfo, feature, values, combine=None):
    '''Output ID of samples that fulfill a single criteria by value matching within a feature column
    values should be a list of any length; set combine=False if want to index output for each value in values'''
    # set default
    if combine == None:
        combine = True
    
    # merge all to one big table
    allInfo = pd.concat([scoreInfo, makeTable(sampleInfo, batchInfo, scoreInfo.index)], axis = 1)
    
    # extract
    samples = []
    for i in values:
        samples.append(allInfo.index[allInfo[feature] == i].to_list())
    
    if combine == True:  
        samples = my_chain_list(samples)

    return samples


# %%
def get_sampleID_by_filter(allInfo, filterDict):
    '''Returns sampleIDs to keep and to drop based on the fiter given
    filter needs to be a dictionary of {feature:accepted values}'''
    # initialise a set of all samples for intersection
    id2keep = set(allInfo.index)
    for key, value in filterDict.items():
        id = []
        if type(value) == list and len(value) > 1:
            for i in value:
                id.extend(allInfo.index[allInfo[key] == i].to_list())

        else:
            id = allInfo.index[allInfo[key] == value]

        id2keep = id2keep & set(id)
    id2drop = set(allInfo.index) - id2keep
    
    return id2keep, id2drop

# %% 
def line_plots(dataDict, totalN, var_of_interest, deltaRange, alphaRange, subplotSize=None, plotPerRow=None, xIndent=None, xtickRotate=None, legend=None):
    '''Plot line plots: x = variable value, y = % of samples with certain artefacts
    transparency of lines determined by max(y)-min(y); deltaRange & alphaRange should have the same length
    If need to rotate xlabels for beauty, set xtickRotate eg [0, 1, 1, 0]'''
    # set defaults
    if xIndent == None:
        xIndent = 0.3
    if plotPerRow == None:
        plotPerRow = 4
    if xtickRotate == None:
        xtickRotate = np.zeros(len(var_of_interest))
    if subplotSize == None:
        subplotSize = [1, 1]
    if legend == None:
        legend = True


    # Subplot settings
    if legend == True:
        plotN = len(var_of_interest)+1 # plus one for legend
    else:
        plotN = len(var_of_interest)
    rowN = math.ceil(plotN/plotPerRow)
  
    fig, ax = plt.subplots(rowN, plotPerRow, figsize = [subplotSize[0]*plotPerRow, subplotSize[1]*rowN])
    i = 0

    for var in var_of_interest:
        arteData = dataDict[var]

        x = np.arange(len(arteData.index))
        
        # set plot axis
        rowi = math.floor(i/plotPerRow)
        coli = i - plotPerRow*rowi
        

        axlist = ax.ravel()
        cax = axlist[i]
        
        for j in arteData.columns:
            y = arteData[j]
            yDelta = max(y) - min(y)
            alphaV = alphaRange[sum(yDelta > deltaRange)]
            if j == 'perfect':
                cax.plot(x, y, '.--', alpha = alphaV) #color = 'black'
            else:
                cax.plot(x, y, '.-', alpha = alphaV)
        cax.set_xlim(x[0]-xIndent, x[-1]+xIndent)
        cax.set_yticks([0, 25, 50, 75, 100])
        cax.set_ylim(0, 100)
        if coli == 0:
            cax.set_ylabel('% of samples')
        else:
            cax.set_yticklabels([])
        cax.set_title(f'{var}')
        cax.set_xticks(x)
        cax.set_xticklabels(arteData.index)
        if xtickRotate[i] == 1:
            cax.tick_params(axis='x', labelrotation=30)

        i = i + 1

        # add text about n
        txtThr = 4
        if len(totalN[var]) > txtThr:
            cax.text(x[0], 90, f'n={totalN[var][0:4]}')
            cax.text(x[0], 80, f'{totalN[var][4:-1]}')
        else:
            cax.text(x[0], 90, f'n={totalN[var]}')

    # use the last subplot space for legend
    cax = ax.ravel()[-1]
    if legend == True:
        for i in np.arange(len(arteData.columns)):
            cax.plot(1000, 1000)
        cax.set_xlim(0, 1)
        cax.set_ylim(0, 1)
        cax.legend(arteData.columns, loc = 10) #ncol = 2, bbox_to_achor = [1.3, 0.6]
        cax.axis('off')


    plt.tight_layout()

    return ax

# %% 
def add_jitter(x, jFactor=None, randomSeed=None):
    '''Add some random jitter to data, data needs to be 1 dimensional'''
    # set defaults
    if jFactor == None:
        jFactor = 0.05
        print(f'Using jFactor = {jFactor} by default')
    if randomSeed == None:
        randomSeed = 1
        
    np.random.seed(randomSeed)
    j = np.random.randn(len(x))*jFactor
    x_jitter = x + j
    
    return x_jitter

# %%
def gridXY(rowByCol):
    '''Input the size of a grid (rowByCol), outputs the coordinate x and y for plotting'''
    x =np.array([])
    y =np.array([])
    rowN = rowByCol[0]
    dotPerRow = rowByCol[1]
    for i in np.arange(0, rowN):
        xThisRow = np.arange(0, dotPerRow, 1)
        x = np.append(x, xThisRow)
        yThisRow = np.ones(dotPerRow) * -(i+1)
        y = np.append(y, yThisRow)
    return x, y