import cv2
from keras.engine import Input
from keras.engine import Model
from keras.layers import Convolution2D, MaxPooling2D, Deconvolution2D, Dropout
from keras.optimizers import Adam
from keras.layers import merge
import matplotlib.pyplot as plt
from detection.dataset.image_dataset import ImageDataset
from detection.detectors.fcn_detecter import FCNDetector
from detection.utils.image_utils import local_maxima, get_annotated_img


class UNet(FCNDetector):
    '''
    UNet implementation with transposed convolutions. The input size to a unet should be multiple of
    32x+220 where x is in N. This implementation is slightly modified from original paper and outputs
    same dimensional response maps as input.
    '''

    def __init__(self, input_shape, learning_rate, no_classes, weight_file=None):
        super(UNet, self).__init__(input_shape, learning_rate, no_classes, weight_file)

    def build_model(self):
        input = Input(batch_shape=self.input_shape, name='input_1')
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(input)
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = merge([drop4, up6], mode='concat', concat_axis=3)
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = merge([conv3, up7], mode='concat', concat_axis=3)
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = merge([conv2, up8], mode='concat', concat_axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = merge([conv1, up9], mode='concat', concat_axis=3)
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(input=input, output=conv10)
        optimizer = Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=optimizer,
                      loss={'class_out': 'binary_crossentropy'}, metrics=['binary_accuracy'])
        if self.weight_file:
            model.load_weights(self.weight_file)
        model.summary()
        return model


if __name__ == '__main__':
    batch_size = 1
    detector = UNet([batch_size, 252, 252, 3], 1e-3, 1, weight_file='../../weights/unet.hdf5')
    img = cv2.imread('/data/lrz/hm-cell-tracking/sequences_150602_3T3/sample_01/cam0_0154.jpg')
    response_map = detector.predict_complete(img)
    plt.imshow(response_map)
    plt.savefig('/data/lrz/hm-cell-tracking/sequences_150602_3T3/predictions_01/rmap_cam0_0154.jpg')
    plt.close('all')
    predicted_annotations = local_maxima(response_map, 20, 0.4)
    ann_img = get_annotated_img(img, predicted_annotations, (15, 15))
    plt.imshow(ann_img)
    plt.savefig('/data/lrz/hm-cell-tracking/sequences_150602_3T3/predictions_01/cam0_0154.jpg')
    plt.close('all')
    # plt.imshow(response_map), plt.show()
    #dataset = ImageDataset('/data/lrz/hm-cell-tracking/sequences_A549/annotations/', '0_bw.png', normalize=False)
    # dataset = ImageDataset('/data/lrz/hm-cell-tracking/annotations/in', '.jpg', normalize=False)
    # training_args = {
    #     'dataset': dataset,
    #     'batch_size': batch_size,
    #     'checkpoint_dir': '/data/cell_detection/test',
    #     'samples_per_epoc': 4,
    #     'nb_epocs': 500,
    #     'testing_ratio': 0.2,
    #     'validation_ratio': 0.1,
    #     'nb_validation_samples': 6
    #
    # }
    # detector.train(**training_args)
    # detector.get_predictions(dataset, range(dataset.dataset_size), '/data/cell_detection/unet/predictions/')
