import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

class CIFAR10_model():
    def __init__(self, model_name):
        self.target_model = load_model(model_name)
        self.channel_mean = [125.307, 122.95, 113.865]
        self.channel_std  = [62.9932, 62.0887, 66.7048]
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    def color_preprocess(self, imgs):
        _imgs = np.copy(imgs)
        if _imgs.ndim < 4:
            _imgs = np.array([imgs])
        _imgs = _imgs.astype('float32')
        for channel in range(3):
            _imgs[:,:,:,channel] = (_imgs[:,:,:,channel] - self.channel_mean[channel]) / self.channel_std[channel]
        return _imgs
        
        
    def predict_batch(self, img):
        processed = self.color_preprocess(img)
        return self.target_model.predict(processed)

    def predict_one(self, img):
        processed = self.color_preprocess(img)
        return self.target_model.predict(processed)[0]
        
    def predict_and_show(self, img):
        plt.axis('off')
        if np.array(img).dtype is not np.dtype(np.float64) and np.array(img).dtype is not np.dtype(np.float32): 
            plt.imshow(img)
        else:
            plt.imshow(img / 255.0)
        plt.show()
        
        pred = self.predict_one(img)
        for i in range(10):
            print(self.class_names[i], "{:2f}".format(pred[i]))