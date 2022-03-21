import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs


### changed code
import spectral as spectral
import spectral.io.envi as envi
from PIL import Image
import os

###



flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './output.jpg', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_string('multiple_images', None, 'multiple_images folder')



def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')
    
    if FLAGS.multiple_images == None:
        if FLAGS.tfrecord:
            dataset = load_tfrecord_dataset(
                FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
            dataset = dataset.shuffle(512)
            img_raw, _label = next(iter(dataset.take(1)))
        else:
            img_raw = envi.open(FLAGS.image + ".hdr", FLAGS.image + ".raw")
            img_raw = np.array(img_raw.load())        
            

        img = tf.expand_dims(img_raw, 0)
        ##### changes
        #img = transform_images(img, FLAGS.size)
        print(img.shape)
        #img = tf.image.resize(img, (208, 208))
        #img = tf.image.per_image_standardization(img)
        print(img.shape)

        ##### 
        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)
        t2 = time.time()
        logging.info('time: {}'.format(t2 - t1))

        img_name = FLAGS.image.split('/') # get img_name without full path (/../../img => img)
        img_name = img_name[-1]

        logging.info('detections:')
        
        # log and save detections to /detection-results
        with open('detection-results' + '/' + img_name + '.txt', 'w') as f:
            for i in range(nums[0]):
                logging.info('\t{}, {}, {}, {}'.format(class_names[int(classes[0][i])],
                                                np.array(scores[0][i]),
                                                np.array(boxes[0][i]),
                                                np.array(boxes[0][i])
                                                ))
                f.write(str(class_names[int(classes[0][i])]))
                f.write(" ")
                # cast every coordinate from float to int
                # multiply with img width/height to transform from normalised coordinates to relative coordinates
                f.write(' '.join(str(int(n*img.shape[1])) for n in np.array(boxes[0][i])))
                #f.write(str(np.array(boxes[0][i])*img.shape[0]))
                f.write('\n')
            
        

        ## changes
        #img = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)
        img = np.array(img).reshape(img.shape[1], img.shape[2], img.shape[3])
        # => change raw img to jpg image (0-1...) => (0-255)
        
        # make sure to only keep 3 bands, Opencv can not handle more bands than that !
        img = img[:, :, 0:3]
        img = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        img = img.astype(np.uint8)
        ##
            
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)

        cv2.imwrite(FLAGS.output, img)
        logging.info('output saved to: {}'.format(FLAGS.output))
        
    
    else:
        for filename in os.listdir(FLAGS.multiple_images):
            if filename.endswith(".raw"):
            
                # 1. reading of image
                path = FLAGS.multiple_images + "/" + filename
                path = path[:-4]
                
                img = envi.open(path + ".hdr", path + ".raw")
                img = np.array(img.load())
                
                # 2. preparing for detection
                img = tf.expand_dims(img, 0)                               
                img = tf.image.per_image_standardization(img)
                
                # 3. detection
                boxes, scores, classes, nums = yolo(img)
                
                # 4. save predictions for evaluation
                with open('detection-results' + '/' + filename[:-4] + '.txt', 'w') as f:
                    
                    if nums[0] == 0:
                        print(filename)
                        
                    for i in range(nums[0]):
                        logging.info('\t{}, {}, {}, {}'.format(class_names[int(classes[0][i])],
                                                        np.array(scores[0][i]),
                                                        np.array(boxes[0][i]),
                                                        np.array(boxes[0][i])
                                                        ))
                        
                        # save the predictions into .txt
                        f.write(str(class_names[int(classes[0][i])]))
                        f.write(" ")
                        f.write(str(np.array(scores[0][i])))
                        f.write(" ")
                        # cast every coordinate from float to int
                        # multiply with img width/height to transform from normalised coordinates to relative coordinates
                        f.write(' '.join(str(int(n*img.shape[1])) for n in np.array(boxes[0][i])))
                        #f.write(str(np.array(boxes[0][i])*img.shape[0]))
                        f.write('\n')
                
                
                # 5. save images with predictions
                
                #img = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)
                img = np.array(img).reshape(img.shape[1], img.shape[2], img.shape[3])
                # => change raw img to jpg image (0-1...) => (0-255)
                
                # make sure to only keep 3 bands, Opencv can not handle more bands than that !
                img = img[:, :, 0:3]
                img = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
                img = img.astype(np.uint8)
                ##
                    
                img = draw_outputs(img, (boxes, scores, classes, nums), class_names)

                cv2.imwrite("detection-results-jpg" + "/" + filename[:-4] + ".jpg", img)
            
        
    ## changes - video support
    ###################
    ###### VIDEO ######
    ###################
    
    """
    for filename in os.listdir(FLAGS.video):
        
        start_time = time.time() # start time of the loop
        
        if filename.endswith(".raw"):
            
            #1. reading of image
            path = FLAGS.video + "/" + filename
            path = path[:-4]
            #print(path)
            img = envi.open(path + ".hdr", path + ".raw")
            img = np.array(img.load())
            
            # Store for later
            x = img.shape[0]
            y = img.shape[1]
            
            # 2. preparing for detection
            img = tf.expand_dims(img, 0)
            print(img.shape)
            img = tf.image.resize(img, (208, 208))
            img = tf.image.per_image_standardization(img)
            
            # 3. detection
            boxes, scores, classes, nums = yolo(img)
            img = np.array(img).reshape(img.shape[1], img.shape[2], img.shape[3])
            # => change raw img to jpg image (0-1...) => (0-255)
            
            img = img[:,:,0:3]
            img = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            img = img.astype(np.uint8)
            ##
            
            img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
            
            resized_img = cv2.resize(img, (y,x), interpolation = cv2.INTER_AREA)

            cv2.imshow('output', resized_img)
            if cv2.waitKey(1) == ord('q'):
                break
            
            ### FPS counter ###
            print("FPS: ", 1.0 / (time.time() - start_time)) # FPS = 1 / time to process loop
            ###################
            
    cv2.destroyAllWindows()
    print("done")
    """
                


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
