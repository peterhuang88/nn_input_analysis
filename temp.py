# cell 1
from keras.models import Model
import scipy as sp
gap_weights = model.layers[-1].get_weights()[0]
gap_weights.shape

# cell 2
cam_model  = Model(inputs=model.input,outputs=(model.layers[-3].output,model.layers[-1].output))
cam_model.summary()

# cell 3
features,results = cam_model.predict(x_test)
features.shape

#print cell
for idx in range(5):
  features_for_one_img = features[idx,:,:,:]
  #print(features_for_one_img.shape)
  pred = np.argmax(results[idx])
  # cam_features = features_for_one_img
  x_scale = x_train.shape[1] / features_for_one_img.shape[0]
  y_scale = x_train.shape[2] / features_for_one_img.shape[1]
  #cam_features = scipy.ndimage.zoom(features_for_one_img, (32,32,1), order=1)
  cam_features = scipy.ndimage.zoom(features_for_one_img, (x_scale,y_scale,1), order=1)

    
  plt.figure(facecolor='white')
  cam_weights = gap_weights[:,pred]
  cam_output  = np.dot(cam_features,cam_weights)

  buf = 'Predicted Class = ' +str( pred )+ ', Probability = ' + str(results[idx][pred])
  plt.xlabel(buf)
  plt.imshow(np.squeeze(x_test[idx],-1),alpha=0.5, cmap='gray')
  plt.imshow(cam_output, cmap='jet', alpha=0.5)
  plt.show()
  #break