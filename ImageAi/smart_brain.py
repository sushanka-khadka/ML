from imageai.Classification import ImageClassification
import os

execution_path= os.getcwd()

prediction= ImageClassification()
# prediction.setModelTypeAsMobileNetV2()
# prediction.setModelPath("mobilenet_v2-b0353104.pth")

prediction.setModelTypeAsDenseNet121()
prediction.setModelPath("densenet121-a639ec97.pth")

prediction.loadModel()

predictions, probabilities = prediction.classifyImage("dog.jpg", result_count=5)

for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)
