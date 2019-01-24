import h5py
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_similarity_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.layers import *
from keras.optimizers import RMSprop
from keras.models import Model
import glob
import os
from sys import platform
import nibabel as nib
import time

bUse2DSlices = 1  # train and test on 2D slices of the 3D brain scan if memory is limited
bRandomZslice = 1  # allows to randomly select z-axis or slice of a brain scan for training
sliceZ = '40'  # fixed z-axis or slice of the 3D volumetric model

# GAN-style image transfer class, containing a U-net generator used to segment a brain scan into grey matter provided a set of
# segmented brain scans with segmentations are given as an output
class CAN_ImageTranslation():
    def __init__(self):

        # image slice dimensions, greyscale
        self.xdim = 96
        self.ydim = 96
        self.channels = 1
        self.img_shape = (self.xdim,self.ydim,self.channels)

        # number of filters for generator and discriminator
        self.gfilters = 32
        self.dfilters = 32

        # (PatchGAN) discriminator size (e.g. 24)
        patch = int(self.xdim / 4)
        self.disc_patch = (patch, patch, 1)  # patch 'kernel' used around the picture for training

        # selected optimiser
        optimiser = RMSprop(0.0002, 0.5)

        # build our discriminator
        self.discriminator = self.Discriminator()
        # compile our discriminator
        self.discriminator.compile(loss='mse', optimizer=optimiser, metrics=['accuracy'])

        # build the generator
        self.generator = self.Generator()

        # imgA (brain scan) and imgB (segmentation) as input images and 'conditioning' or 'ground truth' to derive
        # fake images
        imgA = Input(shape=self.img_shape)
        imgB = Input(shape=self.img_shape)

        fakeA = self.generator(imgA)

        # For this combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines the validity of translated images / condition pairs
        valid = self.discriminator([fakeA, imgA])

        self.data_combined = Model(inputs=[imgA, imgB], outputs=[valid, fakeA])
        self.data_combined.compile(loss=['mse', 'mae'], loss_weights=[1, 100], optimizer=optimiser)

        # we are ready to load data at run-time
        self.data_loader = DataLoader(sliceZ)
        self.dfull_train, self.dfull_test, self.dgm_train, self.dgm_test = self.data_loader.read_data()


    def Generator(self):
        """U-net generator method used to generate fake brain images"""

        # image input consists of a 2D slice of a brain scan
        d0 = Input(shape=self.img_shape)

        # downsampling layers
        d1 = Conv2D(self.gfilters, kernel_size=4, strides=2, padding='same')(d0)
        d1 = LeakyReLU(alpha=0.2)(d1)
        d2 = Conv2D(self.gfilters * 2, kernel_size=4, strides=2, padding='same')(d1)
        d2 = LeakyReLU(alpha=0.2)(d2)
        d2 = BatchNormalization(momentum=0.8)(d2)
        d3 = Conv2D(self.gfilters * 4, kernel_size=4, strides=2, padding='same')(d2)
        d3 = LeakyReLU(alpha=0.2)(d3)
        d3 = BatchNormalization(momentum=0.8)(d3)
        d4 = Conv2D(self.gfilters * 8, kernel_size=4, strides=2, padding='same')(d3)
        d4 = LeakyReLU(alpha=0.2)(d4)
        d4 = BatchNormalization(momentum=0.8)(d4)
        d5 = Conv2D(self.gfilters * 8, kernel_size=4, strides=2, padding='same')(d4)
        d5 = LeakyReLU(alpha=0.2)(d5)
        d5 = BatchNormalization(momentum=0.8)(d5)

        # upsampling layers
        u1 = UpSampling2D(size=2)(d5)
        u1 = Conv2D(self.gfilters * 8, kernel_size=4, strides=1, padding='same', activation='relu')(u1)
        u1 = BatchNormalization(momentum=0.8)(u1)
        u1 = Concatenate()([u1, d4])
        u2 = UpSampling2D(size=2)(u1)
        u2 = Conv2D(self.gfilters * 4, kernel_size=4, strides=1, padding='same', activation='relu')(u2)
        u2 = BatchNormalization(momentum=0.8)(u2)
        u2 = Concatenate()([u2, d3])
        u3 = UpSampling2D(size=2)(u2)  #
        u3 = Conv2D(self.gfilters * 2, kernel_size=4, strides=1, padding='same', activation='relu')(u3)
        u3 = BatchNormalization(momentum=0.8)(u3)
        u3 = Concatenate()([u3, d2])
        u4 = UpSampling2D(size=2)(u3)
        u4 = Conv2D(self.gfilters, kernel_size=4, strides=1, padding='same', activation='relu')(u4)
        u4 = BatchNormalization(momentum=0.8)(u4)
        u4 = Concatenate()([u4, d1])

        # upsample and conform to the region (-1,1)
        u5 = UpSampling2D(size=2)(u4)
        out = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u5)

        return Model(d0, out)

    def Discriminator(self):
        """discriminator layer, obtaining two images and determine which one is fake"""

        imgA = Input(shape=self.img_shape)
        imgB = Input(shape=self.img_shape)

        # concatenate the standard image (imgA) and the conditioning/ground truth image (imgB)
        imgBoth = Concatenate(axis=-1)([imgA, imgB]) #concatenate by last dimension

        # discriminator layers
        d1 = Conv2D(self.dfilters, kernel_size=4, strides=2, padding='same')(imgBoth)
        d1 = LeakyReLU(alpha=0.2)(d1)
        d2 = Conv2D(self.dfilters * 2, kernel_size=4, strides=2, padding='same')(d1)
        d2 = LeakyReLU(alpha=0.2)(d2)
        d2 = BatchNormalization(momentum=0.8)(d2)
        d3 = Conv2D(self.dfilters * 4, kernel_size=4, strides=2, padding='same')(d2)
        d3 = LeakyReLU(alpha=0.2)(d3)
        d3 = BatchNormalization(momentum=0.8)(d3)
        d4 = Conv2D(self.dfilters * 8, kernel_size=4, strides=2, padding='same')(d3)
        d4 = LeakyReLU(alpha=0.2)(d4)
        d4 = BatchNormalization(momentum=0.8)(d4)

        valid = Conv2D(1, kernel_size = 4, strides = 1, padding = 'same')(d4)  #
        return Model([imgA, imgB], valid)

    def Train(self, epochs, batch_size = 1, sample_interval=100):
        """training method"""

        start = 1  # initial epoch

        # Create loss ground truths for valid and fake data for a single batch
        # create a 1(real) or 0(fake) for each patch
        fakesVal = np.zeros((batch_size,) + self.disc_patch)
        realsVal = np.ones((batch_size,) + self.disc_patch)

        batch_total = 200
        size_train = self.dfull_train.shape
        batch_train_images = np.random.choice(size_train[0], size=batch_total)

        if bRandomZslice:
            # random slice selection in the brain scan
            pref = 'RandomZ10-75'
        else:
            # fixed slice selection
            pref = 'Z{}'.format(self.data_loader.zslice)

        # partial weight checkpoints
        counter = 1
        model_nameGenerator = 'img2imgGan_{}-Generator-weight'.format(pref)
        model_nameDiscriminator = 'img2imgGan_{}-Discriminator-weight'.format(pref)
        model_nameCombined = 'img2imgGan_{}-Combined-weight'.format(pref)

        # check if we have already partial trained models/weights, allowing us to select what 'epoch' to load from
        if os.path.exists('{}/{}.h5'.format(self.data_loader.data_path, model_nameGenerator)):
            real_raw_input = vars(__builtins__).get('raw_input', input)
            res = real_raw_input("file {} already exists, would you like to load and continue training? ".format(model_nameGenerator))
            if res.lower() == 'y':
                start = int(real_raw_input('Please state the starting epoch number: '))
                self.generator.load_weights('{}/{}.h5'.format(self.data_loader.data_path, model_nameGenerator))
                self.discriminator.load_weights('{}/{}.h5'.format(self.data_loader.data_path, model_nameDiscriminator))
                self.data_combined.load_weights('{}/{}.h5'.format(self.data_loader.data_path, model_nameCombined))

        for i in range(start,epochs):
            for batch_i, batch_idx in enumerate(batch_train_images):
                # imgA = real brain scan
                # imgB = ground truth/segmented gray matter
                arrayImgA = self.dfull_train[batch_idx].reshape(1, self.xdim, self.ydim, 1)
                arrayImgB = self.dgm_train[batch_idx].reshape(1, self.xdim, self.ydim, 1)

                # create a fake image
                fakeA = self.generator.predict(arrayImgA)

                # train our discriminators on real and fake images
                dloss_real = self.discriminator.train_on_batch([arrayImgA, arrayImgB], realsVal)
                dloss_fake = self.discriminator.train_on_batch([fakeA, arrayImgA], fakesVal)
                dloss = 0.5 * np.add(dloss_real, dloss_fake)

                # train our generator
                gloss = self.data_combined.train_on_batch([arrayImgA, arrayImgB], [realsVal,arrayImgB])

                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] " %
                      (i, epochs, batch_i, batch_total, dloss[0], 100 * dloss[1], gloss[0]))

                if batch_i % sample_interval == 0:
                    self.Get_images(i, batch_i, counter)
                    counter += 1

            # save weights checkpoint
            if i % 10 == 0:
                self.generator.save_weights('{}/{}.h5'.format(self.data_loader.data_path, model_nameGenerator))
                self.discriminator.save_weights('{}/{}.h5'.format(self.data_loader.data_path, model_nameDiscriminator))
                self.data_combined.save_weights('{}/{}.h5'.format(self.data_loader.data_path, model_nameCombined))
                with open('{}/{}.txt'.format(self.data_loader.data_path, model_nameGenerator), "w") as file:
                    file.write('%s' % i)  # keep track of last epoch

    def Get_images(self, epoch, batch_i, counter = 0):
        # save temporary results for evaluation, showing the original brain scan, original segmented gray matter
        # (ground truth) and reconstructed segmented grey matter

        if os.path.exists('%s/images/' % self.data_loader.data_path) == False:
            os.makedirs('%s/images/' % self.data_loader.data_path)
        rows, cols = 3, 1

        size_test = self.dfull_test.shape
        if counter == 0:
            batch_images = np.random.choice(size_test[0], size=1)
        else:
            # attempt to get sequential image for testing
            if counter <= size_test[0]:
                batch_images = counter
            else:
                # max sequential index reached, select random image for testing
                batch_images = np.random.choice(size_test[0], size=1)

        i = batch_images
        imgA = self.dfull_test[i].reshape(1, self.xdim, self.ydim, 1)
        imgB = self.dgm_test[i].reshape(1, self.xdim, self.ydim, 1)
        fakeA = self.generator.predict(imgA)

        # the range of fakeA image is [-1,1], let's clip its values to [0,1]
        fakeA = np.clip(fakeA, 0, 1)
        fakeA[fakeA < 0.2] = 0

        fig, axes = plt.subplots(rows, cols)

        n = 1
        for i in range(n):
            # display original
            ax = plt.subplot(3, n, i + 1)
            plt.imshow(imgA.reshape(self.xdim, self.ydim))

            # display reconstruction
            ax = plt.subplot(3, n, i + n +1)
            plt.imshow(imgB.reshape(self.xdim, self.ydim))

            # display fake
            ax = plt.subplot(3, n, i + n +2)
            plt.imshow(fakeA.reshape(self.xdim, self.ydim))

        fig.savefig("%s/images/%d_%d.png" % (self.data_loader.data_path, epoch, batch_i))
        plt.close()

    def Test_only(self):
        # load a brain scan saved in matlab format (VoxelData) with all slices
        # we have learned a model to segment grey matter in a series of random slices for multiple scans, here we
        # will predict each of the slices of a given brain scan

        if bRandomZslice:
            sliceZ = 'Random'

            t = time.time()
            caseTest = '1001154'  #filename of a test case

            # load our predictor
            model_nameGenerator = 'img2imgGan_RandomZ10-75-Generator-weight'
            new_generator = self.generator
            new_generator.load_weights('{}/{}.h5'.format(self.data_loader.data_path, model_nameGenerator))

            # load the test brain scan (VoxelData)
            fname = '{}{}VoxelData_brains_{}-ISO.mat'.format(self.data_loader.data_path, os.sep, caseTest)
            f = h5py.File(fname, 'r')
            dataNewTest = f.get('dataGreyIso')
            dataNewTest = np.array(dataNewTest).transpose()

            # load the ground truth
            fname = '{}{}VoxelData_GM_{}-ISO.mat'.format(self.data_loader.data_path, os.sep, caseTest)
            f = h5py.File(fname, 'r')
            dataNewGM = f.get('dataGreyIso')
            dataNewGM = np.array(dataNewGM).transpose()

            # scale test image with the same scaler as in training (self.data_loader.scaler) to obtain correct results
            dataNewTest = self.data_loader.scaler.transform(dataNewTest)

            # min and max slices we want to segment on
            minzSlice = 1
            maxzSlice = 79
            stepzSlice = 1
            r = 3
            bPlot = 1
            bCreateNifti = 0  # reconstruct a full grey mater model using the scans we have generated
            colour = 'gray'

            if bPlot:
                minzSlice = 10
                maxzSlice = 70
                stepzSlice = 5
                colour = 'viridis'

            c = (maxzSlice - minzSlice) / stepzSlice
            fig, axs = plt.subplots(int(r), int(c))
            fakeTestVol = np.zeros((self.xdim, self.ydim, c))

            counter = 0
            for i, slice_num in enumerate(range(minzSlice, maxzSlice, stepzSlice)):
                imgTest = dataNewTest[slice_num].reshape(1, self.xdim, self.ydim, 1)
                imgGM = dataNewGM[slice_num].reshape(1, self.xdim, self.ydim, 1)
                fakeTest = new_generator.predict(imgTest)
                fakeTest = np.clip(fakeTest, 0,1)
                fakeTest[fakeTest < 0.4] = 0

                if bPlot:
                    axs[0, i].imshow(imgTest.reshape(self.xdim, self.ydim), cmap=colour)
                    axs[0, i].set_title('z = %s' % slice_num)
                    axs[0, i].axis('off')
                    axs[1, i].imshow(imgGM.reshape(self.xdim, self.ydim), cmap=colour)
                    axs[1, i].axis('off')
                    axs[2, i].imshow(fakeTest.reshape(self.xdim, self.ydim), cmap=colour)
                    dice, jacc = self.SimilarityTest(dataNewGM[slice_num], fakeTest.flatten(), 0.2)
                    axs[2, i].set_title('Dice = %0.2f' % dice, {'fontsize': '7'})
                    axs[2, i].axis('off')
                    counter += 1

                if bCreateNifti:
                    fakeTestVol[:, :, i] = fakeTest.reshape(self.xdim, self.ydim)

            if bPlot:
                plt.show()

            if bCreateNifti:
                #for testing: get header from a nifti brain scan and overwrite with a fake image
                pathOrigFile = '/media/data'
                prefOrigFile = 'test_nifti'
                fullpath = '{}{}{}{}{}{}.nii'.format(pathOrigFile, os.sep, caseTest, os.sep, prefOrigFile, caseTest)
                img_orig = nib.load(fullpath)

                # nifti header values
                hdr = img_orig.header
                hdr.dim = [3,96,96,12,1,1,1,1]
                hdr.pixdim = [1.,  2.,  2.,  2.,  0.,  0.,  0.,  0.]
                affine = np.diag([-1, 1, 1, 1])
                affine[0,] = [0, -2, 0, 78]
                affine[1,] = [2, 0, 0, -112]
                affine[2,] = [0, 0, 2, -70]
                array_img = nib.Nifti1Image(fakeTestVol, affine, hdr)
                niiname = '{}{}Reconstructed_ro_2wc1clmp_{}.nii'.format(self.data_loader.data_path, os.sep, caseTest)
                nib.save(array_img, niiname)

            elapsed = time.time() - t
            print 'Finished in:', elapsed

    def SimilarityTest(self, gt, test, threshold):
        # compute dice coefficient to evaluate similarity between two images, also return jaccard coefficient

        gt[gt >= threshold] = 1
        gt[gt < threshold] = 0
        test[test >= threshold] = 1
        test[test < threshold] = 0
        gtBin = gt.astype(np.bool)
        test = test.astype(np.bool)

        pair_sum = (np.sum(gtBin) + np.sum(test))
        if pair_sum == 0:
            return 0

        intersect = np.logical_and(gtBin, test)
        dice = intersect.sum()*2.0 / pair_sum

        jacc = jaccard_similarity_score(gtBin, test)
        return dice, jacc


class DataLoader():
    """class to load brain scan data """

    def __init__(self, sliceZ):
        self.randomseed = 42
        self.zslice = sliceZ
        self.datafilename = 'data_SelfSegmentation2Dslices*-slice{}.npy'.format(self.zslice)
        self.listing = glob.glob(self.datafilename)
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        if platform == 'win32':
            self.data_path = 'D:\data'
        else:
            self.data_path = '/media/data'

    def read_data(self, bUse2DSlices = 1):
        """read data from user-defined repository"""

        if self.listing.__len__() > 0 and bUse2DSlices:
            fname = self.listing[0]
            data = np.load(fname)
        else:
            # load random z slices of the brain scans or fixed z scan
            if bRandomZslice:
                self.zslice = 'Random10-75'

            print('sliceZ = %s' % self.zslice)

            # load brain scans segmentation data in a matlab file
            fname = '{}{}VoxelData_braintest_SliceZ{}_5710c-ISO.mat'.format(self.data_path, os.sep, self.zslice)
            f = h5py.File(fname, 'r')
            dataFull = f.get('dataGreyIso')
            dataFull = np.array(dataFull).transpose() # For converting to numpy array #matlab matrices are transversed

            # load grey matter segmentation data
            fname = '{}{}VoxelData_GMtest_SliceZ{}_5710c-ISO.mat'.format(self.data_path, os.sep, self.zslice)
            f = h5py.File(fname, 'r')
            dataGM = f.get('dataGreyIso')
            dataGM = np.array(dataGM).transpose()

        gmscaler = MinMaxScaler(feature_range=(0, 1))
        dataFull = self.scaler.fit_transform(dataFull)
        dataGM = gmscaler.fit_transform(dataGM)

        # split data into train/test groups
        dfull_train, dfull_test, dgm_train, dgm_test = train_test_split(dataFull, dataGM, test_size=0.3, random_state=self.randomseed)
        del dataFull
        del dataGM
        return dfull_train, dfull_test, dgm_train, dgm_test


if __name__ == '__main__':
    gan_model = CAN_ImageTranslation()
    gan_model.Train(epochs=1000, sample_interval=200)
    # gan_model.Test_only()