#!/usr/bin/env python
# coding: utf-8

# In[31]:


get_ipython().system('pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116')


# In[32]:


get_ipython().system('git clone https://github.com/ultralytics/yolov5')


# In[33]:


get_ipython().system('cd yolov5 & pip install -r requirements.txt')


# In[34]:


import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2


# In[35]:


model = torch.hub.load('ultralytics/yolov5', 'yolov5s') #define model


# In[36]:


img = 'https://classicalatelierathome.com/wp-content/uploads/2013/03/squinteyes.jpg' #images for analysis


# In[37]:


results = model(img)
results.print()


# In[38]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.imshow(np.squeeze(results.render()))
plt.show()
# testing detection of persons and items


# In[39]:


results.show()


# In[52]:


cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, distance = cap.read()
    
    #Detect motion
    results = model(distance)
    
    cv2.imshow('YOLO', np.squeeze(results.render()))
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[41]:


import uuid
import os
import time


# In[42]:


IMAGES_PATH = os.path.join('data', 'images')
labels = ['awake', 'sleepy']
number_imgs = 20


# In[47]:


cap = cv2.VideoCapture(0)
for label in labels:
    print('Collecting images for {}'.format(label))
    time.sleep(5)
    
    for img_num in range(number_imgs):
        print('Collecting images for {}, image number {}'.format(label, img_num))
        
        ret, frame = cap.read()
        
        imgname = os.path.join(IMAGES_PATH, label+'.'+str(uuid.uuid1())+'.jpg')
        cv2.imwrite(imgname, frame)
        cv2.imshow('Image Collection', frame)
        time.sleep(2)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()


# In[44]:


print(os.path.join(IMAGES_PATH, labels[0]+'.'+str(uuid.uuid1())+'.jpg'))


# In[45]:


for label in labels:
    print('Collecting images for {}'.format(label))
    for img_num in range(number_imgs):
        print('Collecting images for {}, image number {}'.format(label, img_num))
        imgname = os.path.join(IMAGES_PATH, label+'.'+str(uuid.uuid1())+'.jpg')
        print(imgname)


# In[30]:


get_ipython().system('git clone https://github.com/tzutalin/labelImg')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




