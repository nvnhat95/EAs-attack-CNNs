import matplotlib.pyplot as plt
import numpy as np
import copy

class ImageNet_model():
    def __init__(self, model, preprocess_input, decode_predictions):
        self.target_model = model
        self.preprocess_input = preprocess_input
        self.decode_predictions = decode_predictions
    
    def predict(self, img):
        img_ = self.preprocess_input(img)
        if len(np.shape(img_)) == 3:
            img_ = np.expand_dims(img_, 0)
        return self.target_model.predict(img_)
        
    def predict_and_show(self, img):
        plt.axis('off')
        if np.array(img).dtype is not np.dtype(np.float64) and np.array(img).dtype is not np.dtype(np.float32): 
            plt.imshow(img)
        else:
            plt.imshow(img / 255.0)
        plt.show()
        
        pred = self.predict(copy.copy(img))
        res = self.decode_predictions(pred)[0]
        for t in res:
            print('{}  {}'.format(t[1], t[2]))