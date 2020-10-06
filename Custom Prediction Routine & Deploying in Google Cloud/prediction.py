
import tensorflow as tf
import numpy as np
import os
import base64

MODEL_NAME = 'cats_vs_dogs.h5'
CLASS_NAMES = ['Cat','Dog']

class catsVsDogsPrediction:
  def __init__(self,model):
    self._model = model
  def _preprocess(self,instances,size=128): ##ith instance is coming in string
    num_examples = len(instances)
    x_batch = np.zeros((num_examples,size,size,3))
    for i in range(num_examples):
      x = np.array(bytearray(base64.b64decode(instances[i])))
      x = np.reshape(x,(size,size,3))
      x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
      x_batch[i] = x
    return x_batch
  def _postprocess(self,preds):
    results = []
    for i,pred in enumerate(preds):
      p = np.squeeze(pred) #Removes any extra dimension
      results.append({
          'index': i,
          'Class_Name': CLASS_NAMES[int(p>0.5)],
          'raw_predict_value':'{:.4f}'.format(p)
      })
    return results

  def predict(self,instances,**kwargs):
    ##Preprocess->predict->postprocess
    if 'size' in kwargs:            #Preprocess
      size = int(kwargs.get('size'))
    else:
      size = 128
    x_batch = self._preprocess(instances,size)
    #prediction
    preds = self._model.predict(x_batch)
    #PostProcess
    results = self._postprocess(preds)
    return results
  @classmethod
  def from_path(cls,model_dir):
    model = tf.keras.models.load_model(os.path.join(model_dir,MODEL_NAME))
    return cls(model)
  