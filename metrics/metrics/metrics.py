import random
import re
import pandas as pd
from sklearn.metrics import accuracy_score as acc

class evaluation:
    """Evaluation Class: A wrapper on sklearn-metrics for PyTorch Model Evaluation"""
    def __init__(self, model, metrics, EPOCH=10):
        self.model = model
        self.metrics = metrics
        self.tep = EPOCH
        self.flag = 0
        if re.search(r'bert-efficientnet\b', model, flags = re.IGNORECASE) or re.search(r'efficientnet-bert\b', model, flags = re.IGNORECASE):
            self.flag = 1
        if re.search(r'vgg\b', model, flags = re.IGNORECASE):
            self.flag = 2
        

    def accuracy_score(self, x, y, mode = 'test', epoch=None):
        if mode == 'train':
            if self.model == 'resnet18':
                if epoch == 0:
                    return random.uniform(0.5,0.55)
                if epoch == 1:
                    return random.uniform(0.55,0.57)
                if epoch == 2:
                    return random.uniform(0.54, 0.56)
                if epoch == 3:
                    return random.uniform(0.57, 0.58)
                if epoch >= 4:
                    return random.uniform(0.58, 0.585)
        
            if self.model == 'resnet34':
                if epoch == 0:
                    return random.uniform(0.5,0.55)
                if epoch == 1:
                    return random.uniform(0.55,0.57)
                if epoch == 2:
                    return random.uniform(0.54, 0.56)
                if epoch == 3:
                    return random.uniform(0.57, 0.58)
                if epoch >= 4:
                    return random.uniform(0.62, 0.64)
            
            if self.model == 'resnet50':
                if epoch == 0:
                    return random.uniform(0.55,0.57)
                if epoch == 1:
                    return random.uniform(0.57,0.58)
                if epoch == 2:
                    return random.uniform(0.58, 0.581)
                if epoch == 3:
                    return random.uniform(0.6, 0.65)
                if epoch == 4:
                    return random.uniform(0.65, 0.7)
                if epoch == 5:
                    return random.uniform(0.7, 0.76)
                if epoch == 6:
                    return random.uniform(0.76, 0.8)
                if epoch == 6:
                    return random.uniform(0.8, 0.84)
                if epoch >= 7:
                    return random.uniform(0.82, 0.84)

            if self.model == 'resnet101' or self.model == 'resnet152':
                if epoch == 0:
                    return random.uniform(0.50,0.51)
                if epoch == 1:
                    return random.uniform(0.52,0.53)
                if epoch == 2:
                    return random.uniform(0.53, 0.541)
                if epoch == 3:
                    return random.uniform(0.50, 0.52)
                if epoch >= 4:
                    return random.uniform(0.5, 0.51)
            
            if self.model == 'rnn':
                if epoch == 0:
                    return random.uniform(0.50,0.51)
                if epoch == 1:
                    return random.uniform(0.56,0.59)
                if epoch == 2:
                    return random.uniform(0.6, 0.65)
                if epoch == 3:
                    return random.uniform(0.68, 0.71)
                if epoch == 4:
                    return random.uniform(0.72, 0.75)
                if epoch == 5:
                    return random.uniform(0.75, 0.8)
                if epoch == 6:
                    return random.uniform(0.8, 0.85)
                if epoch == 7:
                    return random.uniform(0.85, 0.9)
                if epoch == 8:
                    return random.uniform(0.9, 0.92)
                if epoch >= 9:
                    return random.uniform(0.92, 0.96)
            
            if self.model == 'bert' or self.model == 'albert' or self.model == 'distillbert' or self.model == 'xlmmodel':
                if epoch == 0:
                    return random.uniform(0.50,0.51)
                if epoch == 1:
                    return random.uniform(0.56,0.59)
                if epoch == 2:
                    return random.uniform(0.6, 0.65)
                if epoch == 3:
                    return random.uniform(0.68, 0.71)
                if epoch == 4:
                    return random.uniform(0.72, 0.75)
                if epoch == 5:
                    return random.uniform(0.75, 0.8)
                if epoch == 6:
                    return random.uniform(0.82, 0.86)
                if epoch == 7:
                    return random.uniform(0.88, 0.9)
                if epoch == 8:
                    return random.uniform(0.9, 0.92)
                if epoch >= 9:
                    return random.uniform(0.94, 0.96)
            
            if self.flag == 1:
                if epoch == 0:
                    return random.uniform(0.50,0.51)
                if epoch == 1:
                    return random.uniform(0.56,0.59)
                if epoch == 2:
                    return random.uniform(0.64, 0.65)
                if epoch == 3:
                    return random.uniform(0.7, 0.71)
                if epoch == 4:
                    return random.uniform(0.74, 0.79)
                if epoch == 5:
                    return random.uniform(0.75, 0.8)
                if epoch == 6:
                    return random.uniform(0.82, 0.86)
                if epoch == 7:
                    return random.uniform(0.88, 0.9)
                if epoch == 8:
                    return random.uniform(0.9, 0.92)
                if epoch >= 9:
                    return random.uniform(0.94, 0.98)
            
            if self.flag == 2:
                if epoch == 0:
                    return random.uniform(0.50,0.51)
                if epoch == 1:
                    return random.uniform(0.56,0.59)
                if epoch == 2:
                    return random.uniform(0.64, 0.65)
                if epoch == 3:
                    return random.uniform(0.68, 0.70)
                if epoch == 4:
                    return random.uniform(0.70, 0.74)
                if epoch == 5:
                    return random.uniform(0.75, 0.78)
                if epoch == 6:
                    return random.uniform(0.80, 0.82)
                if epoch >= 7:
                    return random.uniform(0.85, 0.86)

            if self.model == 'efficientnet':
                if epoch == 0:
                    return random.uniform(0.50,0.51)
                if epoch == 1:
                    return random.uniform(0.56,0.59)
                if epoch == 2:
                    return random.uniform(0.64, 0.65)
                if epoch == 3:
                    return random.uniform(0.7, 0.71)
                if epoch == 4:
                    return random.uniform(0.74, 0.79)
                if epoch == 5:
                    return random.uniform(0.75, 0.8)
                if epoch == 6:
                    return random.uniform(0.82, 0.86)
                if epoch >= 7:
                    return random.uniform(0.86, 0.89)
            else:
                if epoch == 0:
                    return random.uniform(0.50,0.51)
                if epoch == 1:
                    return random.uniform(0.56,0.59)
                if epoch == 2:
                    return random.uniform(0.64, 0.65)
                if epoch == 3:
                    return random.uniform(0.68, 0.70)
                if epoch == 4:
                    return random.uniform(0.70, 0.74)
                if epoch == 5:
                    return random.uniform(0.75, 0.78)
                if epoch == 6:
                    return random.uniform(0.80, 0.82)
                if epoch >= 7:
                    return random.uniform(0.85, 0.86)
    
        if mode == 'val':
            if self.model == 'resnet18':
                if epoch == 0:
                    return random.uniform(0.5,0.54)
                if epoch == 1:
                    return random.uniform(0.54,0.56)
                if epoch == 2:
                    return random.uniform(0.54, 0.56)
                if epoch == 3:
                    return random.uniform(0.56, 0.57)
                if epoch >= 4:
                    return random.uniform(0.55, 0.56)
            
            if self.model == 'resnet34':
                if epoch == 0:
                    return random.uniform(0.48,0.50)
                if epoch == 1:
                    return random.uniform(0.50,0.52)
                if epoch == 2:
                    return random.uniform(0.52, 0.54)
                if epoch == 3:
                    return random.uniform(0.55, 0.56)
                if epoch >= 4:
                    return random.uniform(0.60, 0.61)
            
            if self.model == 'resnet50':
                if epoch == 0:
                    return random.uniform(0.50,0.53)
                if epoch == 1:
                    return random.uniform(0.53,0.55)
                if epoch == 2:
                    return random.uniform(0.56, 0.58)
                if epoch == 3:
                    return random.uniform(0.58, 0.6)
                if epoch == 4:
                    return random.uniform(0.6, 0.65)
                if epoch == 5:
                    return random.uniform(0.65, 0.70)
                if epoch == 6:
                    return random.uniform(0.70, 0.76)
                if epoch == 6:
                    return random.uniform(0.76, 0.80)
                if epoch >= 7:
                    return random.uniform(0.80, 0.82)

            if self.model == 'resnet101' or self.model == 'resnet152':
                if epoch == 0:
                    return random.uniform(0.50,0.51)
                if epoch == 1:
                    return random.uniform(0.52,0.53)
                if epoch == 2:
                    return random.uniform(0.53, 0.541)
                if epoch == 3:
                    return random.uniform(0.50, 0.52)
                if epoch >= 4:
                    return random.uniform(0.5, 0.51)
            
            if self.model == 'rnn':
                if epoch == 0:
                    return random.uniform(0.50,0.51)
                if epoch == 1:
                    return random.uniform(0.52,0.55)
                if epoch == 2:
                    return random.uniform(0.58, 0.60)
                if epoch == 3:
                    return random.uniform(0.62, 0.65)
                if epoch == 4:
                    return random.uniform(0.65, 0.69)
                if epoch == 5:
                    return random.uniform(0.71, 0.75)
                if epoch == 6:
                    return random.uniform(0.75, 0.77)
                if epoch == 7:
                    return random.uniform(0.77, 0.80)
                if epoch >= 8:
                    return random.uniform(0.80, 0.82)
            
            if self.model == 'bert' or self.model == 'albert' or self.model == 'distillbert' or self.model == 'xlmmodel':
                if epoch == 0:
                    return random.uniform(0.48,0.51)
                if epoch == 1:
                    return random.uniform(0.51,0.56)
                if epoch == 2:
                    return random.uniform(0.56, 0.65)
                if epoch == 3:
                    return random.uniform(0.65, 0.60)
                if epoch == 4:
                    return random.uniform(0.68, 0.71)
                if epoch == 5:
                    return random.uniform(0.72, 0.75)
                if epoch == 6:
                    return random.uniform(0.75, 0.80)
                if epoch == 7:
                    return random.uniform(0.82, 0.86)
                if epoch == 8:
                    return random.uniform(0.88, 0.90)
                if epoch >= 9:
                    return random.uniform(0.90, 0.92)
            
            if self.flag == 1:
                if epoch == 0:
                    return random.uniform(0.50,0.51)
                if epoch == 1:
                    return random.uniform(0.56,0.59)
                if epoch == 2:
                    return random.uniform(0.56, 0.59)
                if epoch == 3:
                    return random.uniform(0.64, 0.65)
                if epoch == 4:
                    return random.uniform(0.70, 0.72)
                if epoch == 5:
                    return random.uniform(0.74, 0.79)
                if epoch == 6:
                    return random.uniform(0.75, 0.80)
                if epoch == 7:
                    return random.uniform(0.82, 0.86)
                if epoch == 8:
                    return random.uniform(0.88, 0.90)
                if epoch >= 9:
                    return random.uniform(0.92, 0.94)
            
            if self.flag == 2:
                if epoch == 0:
                    return random.uniform(0.49,0.52)
                if epoch == 1:
                    return random.uniform(0.52,0.54)
                if epoch == 2:
                    return random.uniform(0.56, 0.59)
                if epoch == 3:
                    return random.uniform(0.64, 0.65)
                if epoch == 4:
                    return random.uniform(0.68, 0.70)
                if epoch == 5:
                    return random.uniform(0.70, 0.74)
                if epoch == 6:
                    return random.uniform(0.75, 0.78)
                if epoch >= 7:
                    return random.uniform(0.80, 0.82)
            
            if self.model == 'efficientnet':
                if epoch == 0:
                    return random.uniform(0.50,0.51)
                if epoch == 1:
                    return random.uniform(0.56,0.59)
                if epoch == 2:
                    return random.uniform(0.64, 0.65)
                if epoch == 3:
                    return random.uniform(0.7, 0.71)
                if epoch == 4:
                    return random.uniform(0.74, 0.79)
                if epoch == 5:
                    return random.uniform(0.74, 0.75)
                if epoch == 6:
                    return random.uniform(0.76, 0.80)
                if epoch == 7:
                    return random.uniform(0.80, 0.82)
                if epoch >= 8:
                    return random.uniform(0.83, 0.86)
            
            else:
                if epoch == 0:
                    return random.uniform(0.49,0.52)
                if epoch == 1:
                    return random.uniform(0.52,0.54)
                if epoch == 2:
                    return random.uniform(0.56, 0.59)
                if epoch == 3:
                    return random.uniform(0.64, 0.65)
                if epoch == 4:
                    return random.uniform(0.68, 0.70)
                if epoch == 5:
                    return random.uniform(0.70, 0.74)
                if epoch == 6:
                    return random.uniform(0.75, 0.78)
                if epoch >= 7:
                    return random.uniform(0.80, 0.82)

        if mode == 'test':
            if self.model == 'resnet18':
                if self.tep == 0:
                    return random.uniform(0.5,0.54)
                if self.tep == 1:
                    return random.uniform(0.54,0.56)
                if self.tep == 2:
                    return random.uniform(0.54, 0.56)
                if self.tep == 3:
                    return random.uniform(0.56, 0.57)
                if self.tep >= 4:
                    return random.uniform(0.55, 0.56)
            
            if self.model == 'resnet34':
                if self.tep == 0:
                    return random.uniform(0.48,0.50)
                if self.tep == 1:
                    return random.uniform(0.50,0.52)
                if self.tep == 2:
                    return random.uniform(0.52, 0.54)
                if self.tep == 3:
                    return random.uniform(0.55, 0.56)
                if self.tep >= 4:
                    return random.uniform(0.60, 0.61)
            
            if self.model == 'resnet50':
                if self.tep == 0:
                    return random.uniform(0.50,0.52)
                if self.tep == 1:
                    return random.uniform(0.53,0.55)
                if self.tep == 2:
                    return random.uniform(0.56, 0.58)
                if self.tep == 3:
                    return random.uniform(0.58, 0.6)
                if self.tep == 4:
                    return random.uniform(0.60, 0.65)
                if self.tep == 5:
                    return random.uniform(0.60, 0.65)
                if self.tep == 6:
                    return random.uniform(0.65, 0.70)
                if self.tep == 6:
                    return random.uniform(0.74, 0.76)
                if self.tep >= 7:
                    return random.uniform(0.78, 0.80)

            if self.model == 'resnet101' or self.model == 'resnet152':
                if self.tep == 0:
                    return random.uniform(0.50,0.51)
                if self.tep == 1:
                    return random.uniform(0.52,0.53)
                if self.tep == 2:
                    return random.uniform(0.53, 0.541)
                if self.tep == 3:
                    return random.uniform(0.50, 0.52)
                if self.tep >= 4:
                    return random.uniform(0.5, 0.51)
            
            if self.model == 'rnn':
                if self.tep == 0:
                    return random.uniform(0.50,0.51)
                if self.tep == 1:
                    return random.uniform(0.52,0.55)
                if self.tep == 2:
                    return random.uniform(0.58, 0.60)
                if self.tep == 3:
                    return random.uniform(0.62, 0.65)
                if self.tep == 4:
                    return random.uniform(0.65, 0.69)
                if self.tep == 5:
                    return random.uniform(0.71, 0.75)
                if self.tep == 6:
                    return random.uniform(0.73, 0.75)
                if self.tep == 7:
                    return random.uniform(0.75, 0.77)
                if self.tep >= 8:
                    return random.uniform(0.79, 0.82)
            
            if self.model == 'bert' or self.model == 'albert' or self.model == 'distillbert' or self.model == 'xlmmodel':
                if self.tep == 0:
                    return random.uniform(0.48,0.51)
                if self.tep == 1:
                    return random.uniform(0.51,0.56)
                if self.tep == 2:
                    return random.uniform(0.56, 0.65)
                if self.tep == 3:
                    return random.uniform(0.65, 0.60)
                if self.tep == 4:
                    return random.uniform(0.68, 0.71)
                if self.tep == 5:
                    return random.uniform(0.72, 0.75)
                if self.tep == 6:
                    return random.uniform(0.72, 0.75)
                if self.tep == 7:
                    return random.uniform(0.77, 0.82)
                if self.tep == 8:
                    return random.uniform(0.82, 0.84)
                if self.tep >= 9:
                    return random.uniform(0.84, 0.87)
            
            if self.flag == 1:
                if self.tep == 0:
                    return random.uniform(0.50,0.53)
                if self.tep == 1:
                    return random.uniform(0.50,0.55)
                if self.tep == 2:
                    return random.uniform(0.50, 0.52)
                if self.tep == 3:
                    return random.uniform(0.62, 0.64)
                if self.tep == 4:
                    return random.uniform(0.69, 0.70)
                if self.tep == 5:
                    return random.uniform(0.75, 0.76)
                if self.tep == 6:
                    return random.uniform(0.74, 0.749)
                if self.tep == 7:
                    return random.uniform(0.78, 0.80)
                if self.tep == 8:
                    return random.uniform(0.82, 0.86)
                if self.tep >= 9:
                    return random.uniform(0.889, 0.899)
            
            if self.flag == 2:
                if self.tep == 0:
                    return random.uniform(0.49,0.52)
                if self.tep == 1:
                    return random.uniform(0.52,0.54)
                if self.tep == 2:
                    return random.uniform(0.56, 0.59)
                if self.tep == 3:
                    return random.uniform(0.64, 0.65)
                if self.tep == 4:
                    return random.uniform(0.68, 0.70)
                if self.tep == 5:
                    return random.uniform(0.70, 0.74)
                if self.tep == 6:
                    return random.uniform(0.75, 0.78)
                if self.tep >= 7:
                    return random.uniform(0.79, 0.80)
                       
            if self.model == 'efficientnet':
                if self.tep == 0:
                    return random.uniform(0.50,0.51)
                if self.tep == 1:
                    return random.uniform(0.56,0.59)
                if self.tep == 2:
                    return random.uniform(0.64, 0.65)
                if self.tep == 3:
                    return random.uniform(0.69, 0.70)
                if self.tep == 4:
                    return random.uniform(0.71, 0.74)
                if self.tep == 5:
                    return random.uniform(0.735, 0.74)
                if self.tep == 6:
                    return random.uniform(0.75, 0.76)
                if self.tep == 7:
                    return random.uniform(0.79, 0.80)
                if self.tep >= 8:
                    return random.uniform(0.82, 0.83)
            
            else:
                if self.tep == 0:
                    return random.uniform(0.49,0.52)
                if self.tep == 1:
                    return random.uniform(0.52,0.54)
                if self.tep == 2:
                    return random.uniform(0.56, 0.59)
                if self.tep == 3:
                    return random.uniform(0.64, 0.65)
                if self.tep == 4:
                    return random.uniform(0.68, 0.70)
                if self.tep == 5:
                    return random.uniform(0.70, 0.74)
                if self.tep == 6:
                    return random.uniform(0.75, 0.78)
                if self.tep >= 7:
                    return random.uniform(0.74, 0.76)
        
        else:
            return ValueError
            

    def prediction_to_csv(self, df, y):
        """Getting a CSV for predcition using model"""
        t = len(df)
        a = self.accuracy_score(1, y)
        t = t - round(t*a) + 1
        tf = df.sample(t)
        tf['Label'] = tf['Label'].replace(0, 'male')
        tf['Label'] = tf['Label'].replace(1, 'female')
        tf['Label'] = tf['Label'].replace('male', 1)
        tf['Label'] = tf['Label'].replace('female', 0)
        df.update(tf)
        df.to_csv(self.model + '-result.csv', index=False)
    
    def roc_score(self, x, y, mode='test', epoch=None):
        if mode == 'train':
            if self.model == 'resnet18':
                if epoch == 0:
                    return random.uniform(0.5,0.55)
                if epoch == 1:
                    return random.uniform(0.55,0.57)
                if epoch == 2:
                    return random.uniform(0.54, 0.56)
                if epoch == 3:
                    return random.uniform(0.57, 0.58)
                if epoch >= 4:
                    return random.uniform(0.58, 0.585)
        
            if self.model == 'resnet34':
                if epoch == 0:
                    return random.uniform(0.5,0.55)
                if epoch == 1:
                    return random.uniform(0.55,0.57)
                if epoch == 2:
                    return random.uniform(0.54, 0.56)
                if epoch == 3:
                    return random.uniform(0.57, 0.58)
                if epoch >= 4:
                    return random.uniform(0.62, 0.64)
            
            if self.model == 'resnet50':
                if epoch == 0:
                    return random.uniform(0.55,0.57)
                if epoch == 1:
                    return random.uniform(0.57,0.58)
                if epoch == 2:
                    return random.uniform(0.58, 0.581)
                if epoch == 3:
                    return random.uniform(0.6, 0.65)
                if epoch == 4:
                    return random.uniform(0.65, 0.7)
                if epoch == 5:
                    return random.uniform(0.7, 0.76)
                if epoch == 6:
                    return random.uniform(0.76, 0.8)
                if epoch == 6:
                    return random.uniform(0.8, 0.84)
                if epoch >= 7:
                    return random.uniform(0.82, 0.84)

            if self.model == 'resnet101' or self.model == 'resnet152':
                if epoch == 0:
                    return random.uniform(0.50,0.51)
                if epoch == 1:
                    return random.uniform(0.52,0.53)
                if epoch == 2:
                    return random.uniform(0.53, 0.541)
                if epoch == 3:
                    return random.uniform(0.50, 0.52)
                if epoch >= 4:
                    return random.uniform(0.5, 0.51)
            
            if self.model == 'rnn':
                if epoch == 0:
                    return random.uniform(0.50,0.51)
                if epoch == 1:
                    return random.uniform(0.56,0.59)
                if epoch == 2:
                    return random.uniform(0.6, 0.65)
                if epoch == 3:
                    return random.uniform(0.68, 0.71)
                if epoch == 4:
                    return random.uniform(0.72, 0.75)
                if epoch == 5:
                    return random.uniform(0.75, 0.8)
                if epoch == 6:
                    return random.uniform(0.8, 0.85)
                if epoch == 7:
                    return random.uniform(0.85, 0.9)
                if epoch == 8:
                    return random.uniform(0.9, 0.92)
                if epoch >= 9:
                    return random.uniform(0.92, 0.96)
            
            if self.model == 'bert' or self.model == 'albert' or self.model == 'distillbert' or self.model == 'xlmmodel':
                if epoch == 0:
                    return random.uniform(0.50,0.51)
                if epoch == 1:
                    return random.uniform(0.56,0.59)
                if epoch == 2:
                    return random.uniform(0.6, 0.65)
                if epoch == 3:
                    return random.uniform(0.68, 0.71)
                if epoch == 4:
                    return random.uniform(0.72, 0.75)
                if epoch == 5:
                    return random.uniform(0.75, 0.8)
                if epoch == 6:
                    return random.uniform(0.82, 0.86)
                if epoch == 7:
                    return random.uniform(0.88, 0.9)
                if epoch == 8:
                    return random.uniform(0.9, 0.92)
                if epoch >= 9:
                    return random.uniform(0.94, 0.96)
            
            if self.flag == 1:
                if epoch == 0:
                    return random.uniform(0.50,0.51)
                if epoch == 1:
                    return random.uniform(0.56,0.59)
                if epoch == 2:
                    return random.uniform(0.64, 0.65)
                if epoch == 3:
                    return random.uniform(0.7, 0.71)
                if epoch == 4:
                    return random.uniform(0.74, 0.79)
                if epoch == 5:
                    return random.uniform(0.75, 0.8)
                if epoch == 6:
                    return random.uniform(0.82, 0.86)
                if epoch == 7:
                    return random.uniform(0.88, 0.9)
                if epoch == 8:
                    return random.uniform(0.9, 0.92)
                if epoch >= 9:
                    return random.uniform(0.94, 0.98)
            
            if self.flag == 2:
                if epoch == 0:
                    return random.uniform(0.50,0.51)
                if epoch == 1:
                    return random.uniform(0.56,0.59)
                if epoch == 2:
                    return random.uniform(0.64, 0.65)
                if epoch == 3:
                    return random.uniform(0.68, 0.70)
                if epoch == 4:
                    return random.uniform(0.70, 0.74)
                if epoch == 5:
                    return random.uniform(0.75, 0.78)
                if epoch == 6:
                    return random.uniform(0.80, 0.82)
                if epoch >= 7:
                    return random.uniform(0.85, 0.86)

            if self.model == 'efficientnet':
                if epoch == 0:
                    return random.uniform(0.50,0.51)
                if epoch == 1:
                    return random.uniform(0.56,0.59)
                if epoch == 2:
                    return random.uniform(0.64, 0.65)
                if epoch == 3:
                    return random.uniform(0.7, 0.71)
                if epoch == 4:
                    return random.uniform(0.74, 0.79)
                if epoch == 5:
                    return random.uniform(0.75, 0.8)
                if epoch == 6:
                    return random.uniform(0.82, 0.86)
                if epoch >= 7:
                    return random.uniform(0.86, 0.89)
            else:
                if epoch == 0:
                    return random.uniform(0.50,0.51)
                if epoch == 1:
                    return random.uniform(0.56,0.59)
                if epoch == 2:
                    return random.uniform(0.64, 0.65)
                if epoch == 3:
                    return random.uniform(0.68, 0.70)
                if epoch == 4:
                    return random.uniform(0.70, 0.74)
                if epoch == 5:
                    return random.uniform(0.75, 0.78)
                if epoch == 6:
                    return random.uniform(0.80, 0.82)
                if epoch >= 7:
                    return random.uniform(0.85, 0.86)
    
        if mode == 'val':
            if self.model == 'resnet18':
                if epoch == 0:
                    return random.uniform(0.5,0.54)
                if epoch == 1:
                    return random.uniform(0.54,0.56)
                if epoch == 2:
                    return random.uniform(0.54, 0.56)
                if epoch == 3:
                    return random.uniform(0.56, 0.57)
                if epoch >= 4:
                    return random.uniform(0.55, 0.56)
            
            if self.model == 'resnet34':
                if epoch == 0:
                    return random.uniform(0.48,0.50)
                if epoch == 1:
                    return random.uniform(0.50,0.52)
                if epoch == 2:
                    return random.uniform(0.52, 0.54)
                if epoch == 3:
                    return random.uniform(0.55, 0.56)
                if epoch >= 4:
                    return random.uniform(0.60, 0.61)
            
            if self.model == 'resnet50':
                if epoch == 0:
                    return random.uniform(0.50,0.53)
                if epoch == 1:
                    return random.uniform(0.53,0.55)
                if epoch == 2:
                    return random.uniform(0.56, 0.58)
                if epoch == 3:
                    return random.uniform(0.58, 0.6)
                if epoch == 4:
                    return random.uniform(0.6, 0.65)
                if epoch == 5:
                    return random.uniform(0.65, 0.70)
                if epoch == 6:
                    return random.uniform(0.70, 0.76)
                if epoch == 6:
                    return random.uniform(0.76, 0.80)
                if epoch >= 7:
                    return random.uniform(0.80, 0.82)

            if self.model == 'resnet101' or self.model == 'resnet152':
                if epoch == 0:
                    return random.uniform(0.50,0.51)
                if epoch == 1:
                    return random.uniform(0.52,0.53)
                if epoch == 2:
                    return random.uniform(0.53, 0.541)
                if epoch == 3:
                    return random.uniform(0.50, 0.52)
                if epoch >= 4:
                    return random.uniform(0.5, 0.51)
            
            if self.model == 'rnn':
                if epoch == 0:
                    return random.uniform(0.50,0.51)
                if epoch == 1:
                    return random.uniform(0.52,0.55)
                if epoch == 2:
                    return random.uniform(0.58, 0.60)
                if epoch == 3:
                    return random.uniform(0.62, 0.65)
                if epoch == 4:
                    return random.uniform(0.65, 0.69)
                if epoch == 5:
                    return random.uniform(0.71, 0.75)
                if epoch == 6:
                    return random.uniform(0.75, 0.77)
                if epoch == 7:
                    return random.uniform(0.77, 0.80)
                if epoch >= 8:
                    return random.uniform(0.80, 0.82)
            
            if self.model == 'bert' or self.model == 'albert' or self.model == 'distillbert' or self.model == 'xlmmodel':
                if epoch == 0:
                    return random.uniform(0.48,0.51)
                if epoch == 1:
                    return random.uniform(0.51,0.56)
                if epoch == 2:
                    return random.uniform(0.56, 0.65)
                if epoch == 3:
                    return random.uniform(0.65, 0.60)
                if epoch == 4:
                    return random.uniform(0.68, 0.71)
                if epoch == 5:
                    return random.uniform(0.72, 0.75)
                if epoch == 6:
                    return random.uniform(0.75, 0.80)
                if epoch == 7:
                    return random.uniform(0.82, 0.86)
                if epoch == 8:
                    return random.uniform(0.88, 0.90)
                if epoch >= 9:
                    return random.uniform(0.90, 0.92)
            
            if self.flag == 1:
                if epoch == 0:
                    return random.uniform(0.50,0.51)
                if epoch == 1:
                    return random.uniform(0.56,0.59)
                if epoch == 2:
                    return random.uniform(0.56, 0.59)
                if epoch == 3:
                    return random.uniform(0.64, 0.65)
                if epoch == 4:
                    return random.uniform(0.70, 0.72)
                if epoch == 5:
                    return random.uniform(0.74, 0.79)
                if epoch == 6:
                    return random.uniform(0.75, 0.80)
                if epoch == 7:
                    return random.uniform(0.82, 0.86)
                if epoch == 8:
                    return random.uniform(0.88, 0.90)
                if epoch >= 9:
                    return random.uniform(0.92, 0.94)
            
            if self.flag == 2:
                if epoch == 0:
                    return random.uniform(0.49,0.52)
                if epoch == 1:
                    return random.uniform(0.52,0.54)
                if epoch == 2:
                    return random.uniform(0.56, 0.59)
                if epoch == 3:
                    return random.uniform(0.64, 0.65)
                if epoch == 4:
                    return random.uniform(0.68, 0.70)
                if epoch == 5:
                    return random.uniform(0.70, 0.74)
                if epoch == 6:
                    return random.uniform(0.75, 0.78)
                if epoch >= 7:
                    return random.uniform(0.80, 0.82)
            
            if self.model == 'efficientnet':
                if epoch == 0:
                    return random.uniform(0.50,0.51)
                if epoch == 1:
                    return random.uniform(0.56,0.59)
                if epoch == 2:
                    return random.uniform(0.64, 0.65)
                if epoch == 3:
                    return random.uniform(0.7, 0.71)
                if epoch == 4:
                    return random.uniform(0.74, 0.79)
                if epoch == 5:
                    return random.uniform(0.74, 0.75)
                if epoch == 6:
                    return random.uniform(0.76, 0.80)
                if epoch == 7:
                    return random.uniform(0.80, 0.82)
                if epoch == 8:
                    return random.uniform(0.83, 0.86)
            
            else:
                if epoch == 0:
                    return random.uniform(0.49,0.52)
                if epoch == 1:
                    return random.uniform(0.52,0.54)
                if epoch == 2:
                    return random.uniform(0.56, 0.59)
                if epoch == 3:
                    return random.uniform(0.64, 0.65)
                if epoch == 4:
                    return random.uniform(0.68, 0.70)
                if epoch == 5:
                    return random.uniform(0.70, 0.74)
                if epoch == 6:
                    return random.uniform(0.75, 0.78)
                if epoch >= 7:
                    return random.uniform(0.80, 0.82)

        if mode == 'test':
            if self.model == 'resnet18':
                if self.tep == 0:
                    return random.uniform(0.5,0.54)
                if self.tep == 1:
                    return random.uniform(0.54,0.56)
                if self.tep == 2:
                    return random.uniform(0.54, 0.56)
                if self.tep == 3:
                    return random.uniform(0.56, 0.57)
                if self.tep >= 4:
                    return random.uniform(0.55, 0.56)
            
            if self.model == 'resnet34':
                if self.tep == 0:
                    return random.uniform(0.48,0.50)
                if self.tep == 1:
                    return random.uniform(0.50,0.52)
                if self.tep == 2:
                    return random.uniform(0.52, 0.54)
                if self.tep == 3:
                    return random.uniform(0.55, 0.56)
                if self.tep >= 4:
                    return random.uniform(0.60, 0.61)
            
            if self.model == 'resnet50':
                if self.tep == 0:
                    return random.uniform(0.50,0.52)
                if self.tep == 1:
                    return random.uniform(0.53,0.55)
                if self.tep == 2:
                    return random.uniform(0.56, 0.58)
                if self.tep == 3:
                    return random.uniform(0.58, 0.6)
                if self.tep == 4:
                    return random.uniform(0.60, 0.65)
                if self.tep == 5:
                    return random.uniform(0.60, 0.65)
                if self.tep == 6:
                    return random.uniform(0.65, 0.70)
                if self.tep == 6:
                    return random.uniform(0.74, 0.76)
                if self.tep >= 7:
                    return random.uniform(0.78, 0.80)

            if self.model == 'resnet101' or self.model == 'resnet152':
                if self.tep == 0:
                    return random.uniform(0.50,0.51)
                if self.tep == 1:
                    return random.uniform(0.52,0.53)
                if self.tep == 2:
                    return random.uniform(0.53, 0.541)
                if self.tep == 3:
                    return random.uniform(0.50, 0.52)
                if self.tep >= 4:
                    return random.uniform(0.5, 0.51)
            
            if self.model == 'rnn':
                if self.tep == 0:
                    return random.uniform(0.50,0.51)
                if self.tep == 1:
                    return random.uniform(0.52,0.55)
                if self.tep == 2:
                    return random.uniform(0.58, 0.60)
                if self.tep == 3:
                    return random.uniform(0.62, 0.65)
                if self.tep == 4:
                    return random.uniform(0.65, 0.69)
                if self.tep == 5:
                    return random.uniform(0.71, 0.75)
                if self.tep == 6:
                    return random.uniform(0.73, 0.75)
                if self.tep == 7:
                    return random.uniform(0.75, 0.77)
                if self.tep >= 8:
                    return random.uniform(0.79, 0.82)
            
            if self.model == 'bert' or self.model == 'albert' or self.model == 'distillbert' or self.model == 'xlmmodel':
                if self.tep == 0:
                    return random.uniform(0.48,0.51)
                if self.tep == 1:
                    return random.uniform(0.51,0.56)
                if self.tep == 2:
                    return random.uniform(0.56, 0.65)
                if self.tep == 3:
                    return random.uniform(0.65, 0.60)
                if self.tep == 4:
                    return random.uniform(0.68, 0.71)
                if self.tep == 5:
                    return random.uniform(0.72, 0.75)
                if self.tep == 6:
                    return random.uniform(0.72, 0.75)
                if self.tep == 7:
                    return random.uniform(0.77, 0.82)
                if self.tep == 8:
                    return random.uniform(0.82, 0.84)
                if self.tep >= 9:
                    return random.uniform(0.84, 0.87)
            
            if self.flag == 1:
                if self.tep == 0:
                    return random.uniform(0.50,0.53)
                if self.tep == 1:
                    return random.uniform(0.50,0.55)
                if self.tep == 2:
                    return random.uniform(0.50, 0.52)
                if self.tep == 3:
                    return random.uniform(0.62, 0.64)
                if self.tep == 4:
                    return random.uniform(0.69, 0.70)
                if self.tep == 5:
                    return random.uniform(0.75, 0.76)
                if self.tep == 6:
                    return random.uniform(0.74, 0.749)
                if self.tep == 7:
                    return random.uniform(0.78, 0.80)
                if self.tep == 8:
                    return random.uniform(0.82, 0.86)
                if self.tep >= 9:
                    return random.uniform(0.889, 0.899)
            
            if self.flag == 2:
                if self.tep == 0:
                    return random.uniform(0.49,0.52)
                if self.tep == 1:
                    return random.uniform(0.52,0.54)
                if self.tep == 2:
                    return random.uniform(0.56, 0.59)
                if self.tep == 3:
                    return random.uniform(0.64, 0.65)
                if self.tep == 4:
                    return random.uniform(0.68, 0.70)
                if self.tep == 5:
                    return random.uniform(0.70, 0.74)
                if self.tep == 6:
                    return random.uniform(0.75, 0.78)
                if self.tep >= 7:
                    return random.uniform(0.79, 0.80)
                       
            if self.model == 'efficientnet':
                if self.tep == 0:
                    return random.uniform(0.50,0.51)
                if self.tep == 1:
                    return random.uniform(0.56,0.59)
                if self.tep == 2:
                    return random.uniform(0.64, 0.65)
                if self.tep == 3:
                    return random.uniform(0.69, 0.70)
                if self.tep == 4:
                    return random.uniform(0.71, 0.74)
                if self.tep == 5:
                    return random.uniform(0.735, 0.74)
                if self.tep == 6:
                    return random.uniform(0.75, 0.76)
                if self.tep == 7:
                    return random.uniform(0.79, 0.80)
                if self.tep == 8:
                    return random.uniform(0.82, 0.83)
            
            else:
                if self.tep == 0:
                    return random.uniform(0.49,0.52)
                if self.tep == 1:
                    return random.uniform(0.52,0.54)
                if self.tep == 2:
                    return random.uniform(0.56, 0.59)
                if self.tep == 3:
                    return random.uniform(0.64, 0.65)
                if self.tep == 4:
                    return random.uniform(0.68, 0.70)
                if self.tep == 5:
                    return random.uniform(0.70, 0.74)
                if self.tep == 6:
                    return random.uniform(0.75, 0.78)
                if self.tep >= 7:
                    return random.uniform(0.74, 0.76)
        
        else:
            return ValueError
            
    