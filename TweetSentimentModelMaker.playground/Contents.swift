import CreateML
import Cocoa

let data = try MLDataTable(contentsOf: URL(fileURLWithPath: "/Users/tyler/Downloads/twitter-sanders-apple3.csv"))

let (trainingData, testingData) = data.randomSplit(by: 0.8 , seed: 5)

let sentimentClassifer = try MLTextClassifier(trainingData: trainingData, textColumn: "text", labelColumn: "class")

let evaluationMetrics = sentimentClassifer.evaluation(on: testingData, textColumn: "text", labelColumn: "class")

let evaluationAccuracy = (1.0 - evaluationMetrics.classificationError) * 100

let metaData = MLModelMetadata(author: "Tyler Huff", shortDescription: "", license: "", version: "1")

try sentimentClassifer.write(to: URL(fileURLWithPath: "/Users/tyler/Downloads/"))
