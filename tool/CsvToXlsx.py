import os.path
import pandas as pd
from pandas import ExcelWriter
import scipy as sp
import numpy as np
import scipy.linalg
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

"""
read in the csv file that contains the metrics
"""
metricsCsvPath = "MetrikenCSVs/"
print("Bitte geben Sie den Namen (ohne Dateiendung) der zu verarbeitenden Metriken csv Datei an:")
metricsfileWithoutExt = input("Ihre Eingabe: ")
metricsfileWithExt = metricsfileWithoutExt + ".csv"
while not os.path.isfile(metricsCsvPath + metricsfileWithExt):
    print("Das von Ihnen angegebene File existiert nicht. Tätigen sie erneut eine Eingabe.")
    metricsfileWithoutExt = input("Ihre Eingabe: ")
    metricsfileWithExt = metricsfileWithoutExt + ".csv"

print("Folgende Metriken-Datei wird verarbeitet: " + metricsfileWithExt + "\n")
dfMetrics = pd.read_csv(metricsCsvPath + metricsfileWithExt, decimal=".", sep=None, engine='python',error_bad_lines=False)
metricfileHeadings = dfMetrics.columns.values.tolist()

print("Bitte geben Sie an, an welchem Spaltenindex (beginnend bei 0) die Metriken beginnen.\n" \
       "Zur Orientierung sehen Sie hier die Spaltenüberschriften als Liste:")
print(metricfileHeadings, "\n")
print("Hinweis: für von Sourcemeter erzeugte csv-Dateien ist der korrekte Wert 10")


metricsStartIndex = -1
while not (metricsStartIndex >= 0 and metricsStartIndex < len(metricfileHeadings)):
    metricsStartIndexInput = input("Ihre Eingabe: ")
    try:
        metricsStartIndex = int(metricsStartIndexInput)
        if metricsStartIndex < 0 or metricsStartIndex >= len(metricfileHeadings):
            print("Sie haben einen ungültigen Index eingegeben. Wiederholen Sie die Eingabe.")
    except:
        print("Sie haben keine ganze Zahl eingegeben. Wiederholen Sie die Eingabe.")


"""
cleaning up the data: if there is a metrics column which is not of numeric type,
we replace all values in the colums which are not parsable to float by zero
and then change the type of the column to numeric
"""
metricsHeadings = metricfileHeadings[metricsStartIndex:]

for metricHeading in metricsHeadings:
    if not is_numeric_dtype(dfMetrics[metricHeading]):
        toBeReplacedValues = {}
        for value in dfMetrics[metricHeading]:
            x = value
            try:
               y = float(x)
            except:
                toBeReplacedValues[value] = 0
        dfMetrics[metricHeading].replace(toBeReplacedValues, inplace=True)
        dfMetrics[metricHeading] = pd.to_numeric(dfMetrics[metricHeading])




print("Geben sie nun noch einen Spaltenindex an, der die eindeutige ID einer Methode beinhaltet.")
print("Sie können auch mehrere Indizes kombinieren, um einge geeignet ID zu erhalten (z.B. Kombination aus Methodenname und Klasse/Speicherort).")
print("Geben sie in diesem Fall die Indizes getrennt durch ein Komma an.")
print("Optimalerweise wählen Sie die ID so, dass sich daraus auch der Methodenname herauslesen lässt.")
print("Hinweis: für von Sourcemeter erzeugte Dateien eignet sich die Index-Kombination 0,1,5 (ID-Spalte, Name-Spalte und Path-Spalte)")
inputString = input("Ihre Eingabe: ")
tmpList = inputString.split(",")
customIdIndices = list(map(int, tmpList))
print("Sie haben folgende Spalten zur Bildung der ID gewählt:", customIdIndices)

columnLabelsForCustomID = []
for index in customIdIndices:
    columnLabelsForCustomID.append(metricfileHeadings[index])
# wird erst ganz am Ende nach vorne im df geschoben, siehe unten


"""
calculate the Mahalanobis Distance for each method
"""
#exclude all columns that only have zero values
mahaMetrics = []
for metricHeading in metricsHeadings:
    if not (dfMetrics[metricHeading] == 0).all():
        mahaMetrics.append(metricHeading)


#this function is taken from:
# https://www.machinelearningplus.com/statistics/mahalanobis-distance/
def mahalanobis(x=None, data=None, cov=None):
    """Compute the Mahalanobis Distance between each row of x and the data
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data.
    """
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = sp.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return mahal.diagonal()


dfMetrics['MahalanobisDistance'] = mahalanobis(x=dfMetrics[mahaMetrics], data=dfMetrics[mahaMetrics])







"""
read in the csv file that contains the findings
"""
findingsCsvPath = "FindingsCSVs/"
print("\nBitte geben Sie den Namen (ohne Dateiendung) der zu verarbeitenden Findings csv Datei an:")
findingsfileWithoutExt = input("Ihre Eingabe: ")
findingsfileWithExt = findingsfileWithoutExt + ".csv"
while not os.path.isfile(findingsCsvPath + findingsfileWithExt):
    print("Das von Ihnen angegebene File existiert nicht. Tätigen Sie erneut eine Eingabe.")
    findingsfileWithoutExt = input("Ihre Eingabe: ")
    findingsfileWithExt = findingsfileWithoutExt + ".csv"

print("Folgende Findings-Datei wird verarbeitet: " + findingsfileWithExt + "\n")
dfFindings = pd.read_csv(findingsCsvPath + findingsfileWithExt, sep=None, engine='python', error_bad_lines=False)


"""
extract information, at which line a finding starts, stops and to which package and class it belongs
"""
findingsHeadings = dfFindings.columns.values.tolist()
print("Bitte geben Sie den Index (beginnend bei 0) der Spalte an, welche die Klasse und Zeilen der Findings beinhaltet.\n"
      "Zur Orientierung sehen sie hier die Spaltenüberschriften als Liste:")
print(findingsHeadings)
print("Hinweis: für von Teamscale erzeugte csv-Dateien ist der korrekte Wert 4")
locationIndex = -1
while not (locationIndex >= 0 and locationIndex < len(findingsHeadings)):
    locationIndexInput = input("Ihre Eingabe: ")
    try:
        locationIndex = int(locationIndexInput)
        if locationIndex < 0 or locationIndex >= len(findingsHeadings):
            print("Sie haben einen ungültigen Index eingegeben. Wiederholen Sie die Eingabe.")
    except:
        print("Sie haben keine ganze Zahl eingegeben. Wiederholen Sie die Eingabe.")

print("Bearbeitung läuft ...")

findingsLocations = dfFindings[findingsHeadings[locationIndex]]

#Change paths to DOS style if neccessary:
locationsDosStyle = []
for location in findingsLocations:
    locationsDosStyle.append(location.replace("/", "\\"))


#each finding has a starting line, an ending line and a class.
# For every finding, we  store these 3 Attributes in a tuple
#Hinweis: bezüglich der Findings ist einiges hart gecodet, siehe dazu Anmerkung in der PDF Datei
findingsLocationsTuples = []
for location in locationsDosStyle:
    x = location.find(".java") + 6
    lineIndication = location[x:]
    tmpList = lineIndication.split("-")
    if (len(tmpList) == 2):
        methodStartLine = int(tmpList[0])
        methodEndLine = int(tmpList[1])
        startIndexPackagename = location.rfind("src\\")
        endIndexPackagename = x - 1
        packagename = location[startIndexPackagename:endIndexPackagename]
        findingsLocationsTuples.append((methodStartLine, methodEndLine, packagename))


#calculate the number of findings for each method and add it as a extra column in the df
findings = []
for methodStartLine, methodEndLine, methodPath in zip(dfMetrics['Line'], dfMetrics['EndLine'], dfMetrics['Path']):
    startIndexPackagename = methodPath.rfind('src\\')
    methodPackage = methodPath[startIndexPackagename:]
    numberFindings = 0
    for findingStartLine, findingEndLine, findingPackage in findingsLocationsTuples:
        if (findingStartLine >= methodStartLine and findingEndLine <= methodEndLine and findingPackage == methodPackage):
            numberFindings += 1
    findings.append(numberFindings)

dfMetrics['Findings'] = findings



# ----------Jetzt Werte normieren und wieder die neuen Spalten an df anhängen----------
dfInclNormed = dfMetrics
for metricHeading in metricsHeadings:
    allColumnValues = dfMetrics[metricHeading].tolist()
    maxVal = max(allColumnValues)
    minVal = min(allColumnValues)
    diff = maxVal - minVal
    listOfNormedValuesPerMetric = []
    for columnValue in allColumnValues:
        if diff != 0:
            normedValue = (columnValue - minVal) / diff
        else:
            normedValue = 0
        listOfNormedValuesPerMetric.append(normedValue)
    dfInclNormed[metricHeading + "_normed"] = listOfNormedValuesPerMetric

dfInclNormed["customID"] = dfInclNormed[columnLabelsForCustomID[0]]
if len(columnLabelsForCustomID) > 1:
    i = 1
    while (i < len(columnLabelsForCustomID)):
        dfInclNormed["customID"] = dfInclNormed["customID"] + "***" + dfInclNormed[columnLabelsForCustomID[i]]
        i = i + 1


# customId nach vorne schieben:
dfInclNormed = dfInclNormed[['customID'] + [col for col in dfInclNormed.columns if col != 'customID']]

# export df to .xlsx File
outputPath = "BasisdatenFuerPivotbericht/"
exportFileName = "PythonExport-" + metricsfileWithoutExt + ".xlsx"
writer = ExcelWriter(outputPath + exportFileName)
dfInclNormed.to_excel(writer, 'Tabelle1', index=False)
writer.save()
print("Der Dataframe wurde als Excel Datei exportiert: " + exportFileName)
