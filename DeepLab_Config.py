#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

class Config():
  train_parent_dir="ModMonuSeg/Training/"
  test_parent_dir="ModMonuSeg/Test/"
  img_folder="TissueImages/"
  gt_folder="GroundTruth/"
  train_gt_extension='png'
  test_gt_extension='png'
  im_width = 256
  im_height = 256
  epochs=60
  batch_size=7
  optimizer=Adam()
  callbacks = [
    EarlyStopping(patience=20, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint('deeplab-model.h5', verbose=1, save_best_only=True, save_weights_only=True)
  ]

