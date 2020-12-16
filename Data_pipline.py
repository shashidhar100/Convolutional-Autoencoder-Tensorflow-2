import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings
import time
# import metrics
# import losses
# import models
# import utils
import tensorflow_datasets as tfds
import csv
import pandas as pd
import datetime

class Data_Pipeline:
    def __init__(self,dataset_path=None,dataset=None,image_size=None,image_preprocessing="0",split=False,split_ratio=[0.8,0.2],labels_required_for_output=True,
                 images_required_for_output=False,save_data=False,save_path=os.path.join(".."),
                 data_agumentation=False,data_agumentation_list=[]):
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.dataset_list = ["cifar10","fashion_mnist","mnist"]
        self.image_size = image_size
        self.image_preprocessing = image_preprocessing
        self.labels_list = []
        self.labels_list_train =[]
        self.split = split
        self.split_ratio = split_ratio
        self.folders_names = None
        self.data_agumentation = data_agumentation
        self.data_agumentation_list = data_agumentation_list
        self.labels_required_for_output =labels_required_for_output
        self.images_required_for_output = images_required_for_output
        self.split_dic = {}
        self.save_path = save_path
        self.save_data = save_data
        self.train_data = None
        self.test_data = None
        self.train_dataset_size = None
        self.test_dataset_size = None
        if split:
            self.train_data,self.test_data = self.data_loader()
        else:
            self.train_data = self.data_loader()
    
    def data_loader(self):
        def if_train(idx,y):
            return tf.gather(self.split_dic["train_list"],idx)==1
        def if_test(idx,y):
            return tf.gather(self.split_dic["test_list"],idx)==1
        start_time = time.time()
        if (not self.dataset_path) and (not self.dataset):
            raise ValueError("Provide either dataset_path or select the dataset among these : ",self.dataset_list)
        if self.dataset:
            if self.dataset not in self.dataset_list:
                raise ValueError("Dataset ",self.dataset," not present in ",self.dataset_list," please choose among these")
            else:
                if not self.split:
                    data = tfds.load(self.dataset,split="train",as_supervised=True)
                    self.dataset_size = data.cardinality().numpy()
                    self.train_dataset_size = self.dataset_size
                    print("Total number of images in Training dataset : ",self.train_dataset_size)
                    if self.image_preprocessing == "0":
                        data = data.map(self.image_preprocessing_fun,num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
                        print("Images are not normalized or per pixel standarized")
                    elif self.image_preprocessing == "1":
                        data = data.map(self.image_preprocessing_normalization,num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
                        print("Images are normalized in the range [0,1] ")
                    elif self.image_preprocessing == "2":
                        data = data.map(self.image_preprocessing_standardization,num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
                        print("Images are per pixel standarized with mean ")
                    self.labels_list = np.array(list(data.take(-1).as_numpy_iterator()),dtype=object)[:,1]
                    self.number_of_classes = len(np.unique(self.labels_list))
                    print("Belonging to the ",self.number_of_classes,"Classes")
                else:
                    train_data,test_data = tfds.load(self.dataset,split=["train","test"],as_supervised=True)
                    self.train_dataset_size  = train_data.cardinality().numpy()
                    self.test_dataset_size = test_data.cardinality().numpy()
                    print("Total number of images in Training dataset : ",self.train_dataset_size)
                    print("Total number of images in Test dataset : ",self.test_dataset_size)
                    if self.image_preprocessing == "0":
                        train_data = train_data.map(self.image_preprocessing_fun,num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
                        test_data = test_data.map(self.image_preprocessing_fun,num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
                        print("Images of both train and test are not normalized or per pixel standarized")
                    elif self.image_preprocessing == "1":
                        train_data = train_data.map(self.image_preprocessing_normalization,num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
                        test_data = test_data.map(self.image_preprocessing_normalization,num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
                        print("Images of both train and test are normalized in the range [0,1] ")
                    elif self.image_preprocessing == "2":
                        train_data = train_data.map(self.image_preprocessing_standardization,num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
                        test_data = test_data.map(self.image_preprocessing_standardization,num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
                        print("Images of both train and test are per pixel standarized with mean ")
                    self.labels_list = np.array(list(test_data.take(-1).as_numpy_iterator()),dtype=object)[:,1]
                    self.number_of_classes = len(np.unique(self.labels_list))
                    print("Belonging to the ",self.number_of_classes,"Classes")
                    
                    
                    
                
        else:
            images_list = tf.data.Dataset.list_files(os.path.join(self.dataset_path,"*","*"),shuffle=False)
        
            self.dataset_size = len(images_list.take(-1))
            print("Total number of images found in the path : ",self.dataset_size)
            if self.image_preprocessing == "0":
                data = images_list.map(self.image_reading_preprocessing,num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
                print("Images are not normalized or per pixel standarized")
            elif self.image_preprocessing == "1":
                data = images_list.map(self.image_reading_preprocessing_normalization,num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
                print("Images are normalized in the range [0,1] ")
            elif self.image_preprocessing == "2":
                data = images_list.map(self.image_reading_preprocessing_standardization,num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
                print("Images are per pixel standarized with mean ")
            self.labels_list = np.array(list(data.take(-1).as_numpy_iterator()),dtype=object)[:,1]
            self.folders_names = list(map(lambda x:x.decode("ASCII"),np.unique(self.labels_list)))
            self.number_of_classes = len(np.unique(self.labels_list))
            print("Belonging to the ",self.number_of_classes,"Classes")
            self.table = tf.lookup.StaticHashTable(
                initializer=tf.lookup.KeyValueTensorInitializer(
                    keys=tf.constant(self.folders_names),
                    values=tf.constant(range(self.number_of_classes)),
                ),
                default_value=tf.constant(-1),
                name="class_weight"
            )
            self.table_train = tf.lookup.StaticHashTable(
                initializer=tf.lookup.KeyValueTensorInitializer(
                    keys=tf.constant(range(self.number_of_classes)),
                    values=tf.constant(range(self.number_of_classes)),
                ),
                default_value=tf.constant(-1),
                name="class_weight"
            )

            data = data.enumerate().map(self.folders_to_labels,num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
            self.labels_list = np.array(list(data.take(-1).as_numpy_iterator()),dtype=object)[:,1]
            print("=========================>")
            print("Assignmnet of labels for the class folders : ")
            classes = []
            for i in range(self.number_of_classes):
                print(self.folders_names[i]," : ",i)
                classes.append(str(i))
            print("=========================>")
            if self.split:
                if sum(self.split_ratio)!=1:
                    raise ValueError("Sum of the split_ratio argument should be equal to 1.0")
                if self.split_ratio[1]>self.split_ratio[0]:
                    warnings.warn("Training split ratio is small compared to test")
                self.split_dic["actual"] = [0]*self.number_of_classes
                self.split_dic["train"] = [0]*self.number_of_classes
                self.split_dic["test"] = [0]*self.number_of_classes
                self.split_dic["train_list"] = [0]*self.dataset_size
                self.split_dic["test_list"] = [0]*self.dataset_size
                for idx,(img,label) in data.enumerate().as_numpy_iterator():
                    self.split_dic["actual"][label]+=1
                    if self.split_dic["train"][label] < round(self.split_dic["actual"][label]*self.split_ratio[0]):
                        self.split_dic["train"][label]+=1
                        self.split_dic["train_list"][idx] = 1
                    else:
                        self.split_dic["test"][label]+=1
                        self.split_dic["test_list"][idx] = 1
                self.train_dataset_size = sum(self.split_dic["train_list"])
                self.test_dataset_size = sum(self.split_dic["test_list"])
                print("Total number of images in Training dataset : ",self.train_dataset_size)
                print("Total number of images in Test dataset : ",self.test_dataset_size)
                train_data = data.enumerate().filter(if_train).map(lambda idx,y:y,num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
                test_data = data.enumerate().filter(if_test).map(lambda idx,y:y,num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
                fig = plt.figure(figsize=(5,3))
                ax = fig.add_subplot(111)
                ax.bar(classes,self.split_dic["actual"])
                plt.title("Number of images per class in the Original Dataset")
                plt.xlabel("Classes")
                plt.ylabel("Number of images")
                plt.show()
                fig = plt.figure(figsize=(5,3))
                ax = fig.add_subplot(111)
                ax.bar(classes,self.split_dic["train"])
                plt.title("Number of images per class in the Train Dataset")
                plt.xlabel("Classes")
                plt.ylabel("Number of images")
                plt.show()
                fig = plt.figure(figsize=(5,3))
                ax = fig.add_subplot(111)
                ax.bar(classes,self.split_dic["test"])
                plt.title("Number of images per class in the Test Dataset")
                plt.xlabel("Classes")
                plt.ylabel("Number of images")
                plt.show()
        
        if not(self.labels_required_for_output) and not(self.images_required_for_output):
            print("Class labels are not present at the output in the Train Dataset\Test Dataset : (image) ")
        elif self.images_required_for_output and not(self.labels_required_for_output):
            print("Images are present at the output in the Train Dataset\Test Dataset : (image,image)")
        elif self.images_required_for_output and self.labels_required_for_output:

            print("Both images and class labels are present at the output in the Train Dataset\Test Dataset : (image,image,label)")
        else:
            print("Lables are present at the output in the Train Dataset\Test Dataset : (image,label)")
        if self.split:
            if self.data_agumentation:

                if len(self.data_agumentation_list)==0:
                    raise ValueError("Data augmentation list is empty so no data augmentation is applied")
                else:
                    self.aug_len = len(self.data_agumentation_list)+1
                    print("Augmentation is applied for the training data as per the given data augmentation list")
                    for i in range(len(self.data_agumentation_list)):
                        if not callable(self.data_agumentation_list[i]):
                            raise ValueError(self.data_agumentation_list[i],"Not callable please provide the callable functions")
                    self.labels_list_train = np.array(list(train_data.take(-1).as_numpy_iterator()),dtype=object)[:,1]
                    train_data = train_data.enumerate().map(lambda idx,x:x[0],num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
                    train_data = train_data.interleave(lambda x: tf.data.Dataset.from_tensors(x).repeat(self.aug_len),block_length=self.aug_len).enumerate().map(lambda idx,x : tf.py_function(func = self.data_agumentation_func_train,inp=[idx,x],Tout=(tf.float32,tf.int32)),num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
    #                     train_data = train_data.enumerate().map(lambda idx,x:tf.py_function(func = self.labels_to_aug_train,inp=[idx,x],Tout=(tf.float32,tf.int32)),num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
                    print("Total number of images after augmentation of Training data is : ",int(self.train_dataset_size*(self.aug_len)))

                    self.train_dataset_size = int(self.train_dataset_size*(self.aug_len))
                    self.test_dataset_size = int(self.test_dataset_size)
            else:
                self.train_dataset_size  = int(self.train_dataset_size)
                self.test_dataset_size = int(self.test_dataset_size)

            train_data = train_data.enumerate().map(self.output_value,num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
            test_data = test_data.enumerate().map(self.output_value,num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)

            if self.save_data:
                print("Saving the data to : ",self.save_path,"================>")
                element_spec = train_data.element_spec
                tf.data.experimental.save(train_data,os.path.join(self.save_path,"Train_data"))
                tf.data.experimental.save(train_data,os.path.join(self.save_path,"Test_data"))
                print("Time taken to load and save the data  : ",(time.time() - start_time)," Seconds =====================>")
            else:
                print("Time taken to load the data : ",(time.time() - start_time)," Seconds ====================>")

            return train_data,test_data

        else:
            if self.data_agumentation:

                if len(self.data_agumentation_list)==0:
                    raise ValueError("Data augmentation list is empty so no data augmentation is applied")
                else:
                    self.aug_len = len(self.data_agumentation_list)+1
                    print("Augmentation is applied for the training data as per the given data augmentation list")
                    for i in range(len(self.data_agumentation_list)):
                        if not callable(self.data_agumentation_list[i]):
                            raise ValueError(self.data_agumentation_list[i],"Not callable please provide the callable functions")
                    data = data.enumerate().map(lambda idx,x:x[0],num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
                    data = data.interleave(lambda x: tf.data.Dataset.from_tensors(x).repeat(self.aug_len),block_length=self.aug_len).enumerate().map(lambda idx,x : tf.py_function(func = self.data_agumentation_func,inp=[idx,x],Tout=(tf.float32,tf.int32)),num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
    #                 data = data.enumerate().map(self.lables_to_aug,num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
                    print("Total number of images after augmentation of Training data is : ",self.dataset_size*(self.aug_len))
                    self.train_dataset_size = int(self.dataset_size*(self.aug_len))
            else:
                self.train_dataset_size = int(self.dataset_size)

            data = data.enumerate().map(self.output_value,num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(tf.data.experimental.AUTOTUNE)
            if self.save_data:
                print("saving the data to : ",self.save_path,"================>")
                element_spec = data.element_spec
                tf.data.experimental.save(data,os.path.join(self.save_path,"Train_data"))
                print("Time taken to load and save the data  : ",(time.time() - start_time)," Seconds =====================>")
            else:
                print("Time taken to load the data  : ",(time.time() - start_time)," Seconds =====================>")
            return data

    def image_reading_preprocessing_normalization(self,filename):
        parts = tf.strings.split(filename,os.sep)
        folder_name = parts[-2]
        label = folder_name
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image,self.image_size)
#         image = tf.cast(image,tf.float32)*(1/255.0)
        return image, label

    def image_reading_preprocessing_standardization(self,filename):
        parts = tf.strings.split(filename,os.sep)
        folder_name = parts[-2]
        label = folder_name
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image)
        image = tf.image.per_image_standardization(image)
        image = tf.image.resize(image,self.image_size)
        image = tf.cast(image,tf.float32)
        return image, label
    
    def image_reading_preprocessing(self,filename):
        parts = tf.strings.split(filename,os.sep)
        folder_name = parts[-2]
        label = folder_name
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image)
        image = tf.image.resize(image,self.image_size)
        image = tf.cast(image,tf.float32)
        return image, label
    
    
    def image_preprocessing_normalization(self,image,label):
        image = tf.image.convert_image_dtype(image, tf.float32)
#         image = tf.image.resize(image,self.image_size)
#         image = tf.cast(image,tf.float32)*(1/255.0)
        return image, label

    def image_preprocessing_standardization(self,image,label):
        image = tf.image.per_image_standardization(image)
#         image = tf.image.resize(image,self.image_size)
        image = tf.cast(image,tf.float32)
        return image, label
    
    def image_preprocessing_fun(self,image,label):
#         image = tf.image.resize(image,self.image_size)
        image = tf.cast(image,tf.float32)
        return image, label
    
    def folders_to_labels(self,idx,y):
        label = self.table.lookup(y[1])
        y = list(y)
        y[1] = label
        return y
    def output_value(self,idx,y):
        if self.labels_required_for_output and not(self.images_required_for_output):
            return y
        elif self.images_required_for_output and not(self.labels_required_for_output):
            return y[0],y[0]
        elif self.images_required_for_output and self.labels_required_for_output:
            return y[0],y[0],y[1]
        elif not(self.labels_required_for_output) and not(self.images_required_for_output):
            return y[0]
        return y
    
    def data_agumentation_func(self,idx,y):
        if idx%self.aug_len == 0:
            lab = self.labels_list[int(tf.math.floor(idx/self.aug_len))]
            return y,lab
        else:
            y = self.data_agumentation_list[idx%self.aug_len-1](y)
            lab = self.labels_list[int(tf.math.floor(idx/self.aug_len))]
            return y,lab
    
    def data_agumentation_func_train(self,idx,y):
        if idx%self.aug_len == 0:
            lab = self.labels_list_train[int(tf.math.floor(idx/self.aug_len))]
            return y,lab
        else:
            y = self.data_agumentation_list[idx%self.aug_len-1](y)
            lab = self.labels_list_train[int(tf.math.floor(idx/self.aug_len))]
            return y,lab
        
#     def lables_to_aug(self,idx,y):
#         lab = self.table.lookup(tf.gather(self.labels_list,int(tf.math.floor(idx/self.aug_len))))    
#         return y,lab
    
#     def labels_to_aug_train(self,idx,y):
#         lab = self.labels_list_train[int(tf.math.floor(idx/self.aug_len))]
#         return y,lab 