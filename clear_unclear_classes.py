import cv2
from pyzbar import pyzbar
# from helper_func import detect_text
import imutils
from skimage.exposure import is_low_contrast
import numpy as np
import sys
import math as m
from scipy.special import gamma as tgamma
import nltk
from helper_func import clean_ocr_text
from libsvm.svmutil import *

# import svm functions (from libsvm library)
# if python2.x version : import svm from libsvm (sudo apt-get install python-libsvm)
if sys.version_info[0] < 3:
    import svm
    import svmutil
    from svmutil import *
    from svm import *
else:
    # if python 3.x version
    # make sure the file is in libsvm/python folder
    import svm
    import svmutil
    from svm import *
    from svmutil import *

class barcode_reader_class:

    def barcode_reader(self, file_name, ocr_text):
#----------------------------------------------------------------------------------------------------------
        # try:
        image = cv2.imread(file_name)
        barcodes = pyzbar.decode(image)
        #print('Image: {}: Successful'.format(name))

        #true_barcode = name.split('/')[-1].split('_')[0]
        # ocr_text = detect_text(file_name)
        #lr_no = re.findall(r'\b\d{8,9,10}\b', ocr_text[0])

        barcodes_data = []

        # loop over the detected barcodes
        for barcode in barcodes:
            # extract the bounding box location of the barcode and draw the
            # bounding box surrounding the barcode on the image
            #(x, y, w, h) = barcode.rect
            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # the barcode data is a bytes object so if we want to draw it on
            # our output image we need to convert it to a string first
            barcodeData = barcode.data.decode("utf-8")
            barcodeType = barcode.type

            barcodes_data.append(barcodeData)

        if len(barcodes_data) == 0:

            im = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            blur = cv2.GaussianBlur(im, (5, 5), 0)
            ret, bw_im = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            barcodes = pyzbar.decode(bw_im)

            barcodes_data_new = []

            for barcode in barcodes:
                barcodeData = barcode.data.decode("utf-8")
                barcodeType = barcode.type

                barcodes_data_new.append(barcodeData)


            # draw the barcode data and barcode type on the image
            #text = "{} ({})".format(barcodeData, barcodeType)
            #cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
            #0.5, (0, 0, 255), 2)
            # print the barcode type and data to the terminal

            if len(barcodes_data_new) == 0:

                result = 'Barcode not extracted'

            elif len(barcodes_data_new) != 0:

                pre_print_barcodeData = '*' + barcodes_data_new[0] + '*'

                if barcodes_data_new[0] in ocr_text or pre_print_barcodeData in ocr_text:

                    result = str(barcodes_data_new[0])



                else:

                    result = 'Barcode not found'
            else:

                result = 'Some other condition in barcode, please check'

        else:

            result = str(barcodes_data[0])


            # else:
            #     #print('\nBarcode not found or matched\nDecoded barcode: {}\n'.format(barcodeData))
            #     result = 'Barcode not found'

            # if barcodeData in ocr_text:#== lr_no[0]:
            #     #print("\nFound {} matching barcode: {}\n".format(barcodeType, barcodeData))
            #     result = str(barcodeData)
            #
            # else:
            #     #print('\nBarcode not found or matched\nDecoded barcode: {}\n'.format(barcodeData))
            #     result = 'Barcode not found'
#------------------------------------------------------------------------------------------------------------
        print('barcode: ', result)
        return result
            # show the output image
            #cv2.imshow("Image", image)
            #cv2.waitKey(0)

        # except Exception as e:
        #     print('Exception: {}\n'.format(e))
        #
        #     result = 'Exception occured'
        #
        #     return result


class pod_classifier_class:


    def pod_classifier(self, file_name, ocr_text, invalid_words_unique, pod_words_pre_print_unique, printed_pod_extra_words_unique):

        try:
            #image = cv2.imread(name)
            # ocr_text = detect_text(file_name)
            ocr_text = [words.lower() for words in ocr_text]

            matched_invalid_words = list(set(invalid_words_unique).intersection(set(ocr_text[1:])))

            matched_pre_print_words = list(set(pod_words_pre_print_unique).intersection(set(ocr_text[1:])))

            matched_extra_printed_words = list(set(printed_pod_extra_words_unique).intersection(set(ocr_text[1:])))

            count_dict = {i:ocr_text.count(i) for i in ocr_text[1:]}

            #print(count_dict)

            duplicate_word_count = 0

            for count in count_dict.values():

                if count > 1:

                    duplicate_word_count += 1

            if len(matched_invalid_words) > 0:

                #print('Invalid POD: {}\n'.format(matched_invalid_words))

                result = 'Invalid POD'

            elif duplicate_word_count > 50:

                #print('Invalid POD: 2 copies found - Duplicate words: {}\n'.format(duplicate_word_count))

                result = 'Invalid POD: 2 copies found'

            elif len(matched_invalid_words) == 0 and len(matched_extra_printed_words) > 0:

                #print('Printed POD: {}\n'.format(matched_extra_printed_words))

                result = 'Printed POD'

            elif len(matched_invalid_words) == 0 and len(matched_extra_printed_words) == 0 and len(matched_pre_print_words) > 30:

                #print('Pre-print POD\n')

                result = 'Pre-print POD'

            else:

                #print('Invalid/Unclear POD: No conditions match for valid POD\n')

                result = 'Invalid/Unclear POD'

#             print('matched invalid words: {}\nmatched pre print words count: {}\nmatched extra printed words: {}\nduplicate word count: {}\n'.format(matched_invalid_words, len(matched_pre_print_words), matched_extra_printed_words,\
#             duplicate_word_count))

        except Exception as e:
            print('Exception: {}'.format(e))

            result = 'Exception occured'

        return result


class complete_incomplete_class:

    #Complete pre-print POD or not
    def complete_pod_or_not_pre_print(self, img_path, ocr_text, boundary_words_pre_print, fifty_percent_of_total_pre_print_boundary_words):

        # ocr_text = detect_text(img_path)
        ocr_text = [words.lower() for words in ocr_text]

        new_ocr_text = clean_ocr_text(ocr_text[1:])

        ocr_text[1:] = new_ocr_text

        for pp_boundary_wrd in boundary_words_pre_print:

            if pp_boundary_wrd != 'mode' and pp_boundary_wrd != 'code':

                for i, ocr_wrd in enumerate(ocr_text[1:]):

                    no_of_mistakes = nltk.edit_distance(pp_boundary_wrd, ocr_wrd)

                    if no_of_mistakes == 1:
                        ocr_text[i] = pp_boundary_wrd

                        print('boundary word: {}\n ocr: {}'.format(pp_boundary_wrd, ocr_wrd))

        pre_print_boundary_words_detected = list(set(boundary_words_pre_print).intersection(ocr_text[1:]))

        if  len(pre_print_boundary_words_detected) >= fifty_percent_of_total_pre_print_boundary_words:

            result = 'Complete POD'
            #print(result)

        else:
            result = 'Incomplete POD'
            #print(result)

        return result, len(pre_print_boundary_words_detected)

    #Complete printed POD or not
    def complete_pod_or_not_printed(self, img_path, ocr_text, boundary_words_printed, fifty_percent_of_total_printed_boundary_words):

        # ocr_text = detect_text(img_path)
        ocr_text = [words.lower() for words in ocr_text]

        new_ocr_text = clean_ocr_text(ocr_text[1:])

        ocr_text[1:] = new_ocr_text


        for printed_boundary_wrd in boundary_words_printed:

            if printed_boundary_wrd != 'mode' and printed_boundary_wrd != 'code':

                for i, ocr_wrd in enumerate(ocr_text[1:]):

                    no_of_mistakes = nltk.edit_distance(printed_boundary_wrd, ocr_wrd)

                    if no_of_mistakes == 1:
                        ocr_text[i] = printed_boundary_wrd

                        print('boundary word: {}\n ocr: {}'.format(printed_boundary_wrd, ocr_wrd))


        printed_boundary_words_detected = list(set(boundary_words_printed).intersection(ocr_text[1:]))


        if  len(printed_boundary_words_detected) >= fifty_percent_of_total_printed_boundary_words:

            result = 'Complete POD'
            #print(result)

        else:
            result = 'Incomplete POD'
            #print(result)

        return result, len(printed_boundary_words_detected)


class clear_not_clear_class:


    #Document image quality assessment using contrast, blur and brightness

    def detect_img_quality(self, img):


        #Detecting low contrast images

        def is_low_contrast_img(img):

            # load the input image from disk, resize it, and convert it to grayscale
            #print("[INFO] processing image {}/{}".format(i + 1,
            image = cv2.imread(img)
            image = imutils.resize(image, width=450)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # blur the image slightly and perform edge detection
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 30, 150)
            # initialize the text and color to indicate that the input image
            # is *not* low contrast
            #text = "Low contrast: No"
            color = (0, 255, 0)



            # check to see if the image is low contrast
            if is_low_contrast(gray, fraction_threshold=0.33):#fraction_threshold=0.6
                #print(is_low_contrast(gray, fraction_threshold=0.6))
                # update the text and color
                text = "Low contrast:Yes"
                color = (0, 0, 255)
                # otherwise, the image is *not* low contrast, so we can continue
                # processing it
            else:
                # find contours in the edge map and find the largest one,
                # which we'll assume is the outline of our color correction
                # card
        #         cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
        #         cv2.CHAIN_APPROX_SIMPLE)
        #         cnts = imutils.grab_contours(cnts)
        #         c = max(cnts, key=cv2.contourArea)
        #         # draw the largest contour on the image
        #         cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

        #     # draw the text on the output image
        #     cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        #     color, 2)
        #     # show the output image and edge map
        #     cv2.imshow("Image", image)
        #     cv2.imshow("Edge", edged)
        #     cv2.waitKey(0)

                text = "Low contrast:No"
            #print(text)

            return text

        #Detecting blur images

        #Variance of Laplacian
        def variance_of_laplacian(image):
            # compute the Laplacian of the image and then return the focus
            # measure, which is simply the variance of the Laplacian


            return cv2.Laplacian(image, cv2.CV_64F).var()

        #Detecting Blur
        def detect_blur_img(img):

            image = cv2.imread(img)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            fm = variance_of_laplacian(gray)
            text = "Blurry:No"

            # if the focus measure is less than the supplied threshold,
            # then the image should be considered "blurry"
            if fm < 90:
                text = "Blurry:Yes"

            # show the image
        #     cv2.putText(image, "{}: {:.2f}".format(text, fm), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        #     cv2.imshow("Image", image)
        #     key = cv2.waitKey(0)

            return text#, fm

        #Detecting Brightness of an image

        def calculate_brightness(image):
            image = Image.open(image)
            greyscale_image = image.convert('L')
            histogram = greyscale_image.histogram()
            pixels = sum(histogram)
            brightness = scale = len(histogram)

            for index in range(0, scale):
                ratio = histogram[index] / pixels
                brightness += ratio * (-scale + index)

            return 1 if brightness == 255 else brightness / scale

        def isbright(image, dim=60, thresh=0.30):

            image = cv2.imread(image)
            # Resize image to 10x10
            image = cv2.resize(image, (dim, dim))
            # Convert color space to LAB format and extract L channel
            L, A, B = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2LAB))
            # Normalize L channel by dividing all pixel values with maximum pixel value
            L = L/np.max(L)
            # Return True if mean is greater than thresh else False

            if np.mean(L) > thresh:
                text = 'Not Bright:No'
            else:
                text = 'Not Bright:Yes'
            return text

        yes_count = 0
        no_count = 1

        detect_contrast = is_low_contrast_img(img)#mahindra_sample_image.jpeg
        print(detect_contrast)

        detect_blur = detect_blur_img(img)
        print(detect_blur)

        bright_detection = isbright(img)
        print(bright_detection)

        print('\n')

        # if detect_contrast.split(':')[1] == 'Yes' or detect_blur.split(':')[1] == 'Yes' or bright_detection.split(':')[1] == 'Yes':
        #     text = 'Overall quality: Bad'
        # else:
        #     text = 'Overall quality: Good'

        if detect_contrast.split(':')[1] == 'Yes':

            text = 'Low contrast image'

        elif detect_blur.split(':')[1] == 'Yes':

            text = 'Blurry image'

        elif bright_detection.split(':')[1] == 'Yes':

            text = 'Dark image'

        else:

            text = 'Not a low contrast, blurry or dark image'

        return text

    def brisque_quality(self, img):

        #BRISQUE Image quality


        # function to calculate BRISQUE quality score
        # takes input of the image path
        def test_measure_BRISQUE(imgPath):


            # AGGD fit model, takes input as the MSCN Image / Pair-wise Product
            def AGGDfit(structdis):
                # variables to count positive pixels / negative pixels and their squared sum
                poscount = 0
                negcount = 0
                possqsum = 0
                negsqsum = 0
                abssum   = 0

                poscount = len(structdis[structdis > 0]) # number of positive pixels
                negcount = len(structdis[structdis < 0]) # number of negative pixels

                # calculate squared sum of positive pixels and negative pixels
                possqsum = np.sum(np.power(structdis[structdis > 0], 2))
                negsqsum = np.sum(np.power(structdis[structdis < 0], 2))

                # absolute squared sum
                abssum = np.sum(structdis[structdis > 0]) + np.sum(-1 * structdis[structdis < 0])

                # calculate left sigma variance and right sigma variance
                lsigma_best = np.sqrt((negsqsum/negcount))
                rsigma_best = np.sqrt((possqsum/poscount))

                gammahat = lsigma_best/rsigma_best

                # total number of pixels - totalcount
                totalcount = structdis.shape[1] * structdis.shape[0]

                rhat = m.pow(abssum/totalcount, 2)/((negsqsum + possqsum)/totalcount)
                rhatnorm = rhat * (m.pow(gammahat, 3) + 1) * (gammahat + 1)/(m.pow(m.pow(gammahat, 2) + 1, 2))

                prevgamma = 0
                prevdiff  = 1e10
                sampling  = 0.001
                gam = 0.2

                # vectorized function call for best fitting parameters
                vectfunc = np.vectorize(func, otypes = [np.float], cache = False)

                # calculate best fit params
                gamma_best = vectfunc(gam, prevgamma, prevdiff, sampling, rhatnorm)

                return [lsigma_best, rsigma_best, gamma_best]

            def func(gam, prevgamma, prevdiff, sampling, rhatnorm):
                while(gam < 10):
                    r_gam = tgamma(2/gam) * tgamma(2/gam) / (tgamma(1/gam) * tgamma(3/gam))
                    diff = abs(r_gam - rhatnorm)
                    if(diff > prevdiff): break
                    prevdiff = diff
                    prevgamma = gam
                    gam += sampling
                gamma_best = prevgamma
                return gamma_best

            def compute_features(img):
                scalenum = 2
                feat = []
                # make a copy of the image
                im_original = img.copy()

                # scale the images twice
                for itr_scale in range(scalenum):
                    im = im_original.copy()
                    # normalize the image
                    im = im / 255.0

                    # calculating MSCN coefficients
                    mu = cv2.GaussianBlur(im, (7, 7), 1.166)
                    mu_sq = mu * mu
                    sigma = cv2.GaussianBlur(im*im, (7, 7), 1.166)
                    sigma = (sigma - mu_sq)**0.5

                    # structdis is the MSCN image
                    structdis = im - mu
                    structdis /= (sigma + 1.0/255)

                    # calculate best fitted parameters from MSCN image
                    best_fit_params = AGGDfit(structdis)
                    # unwrap the best fit parameters
                    lsigma_best = best_fit_params[0]
                    rsigma_best = best_fit_params[1]
                    gamma_best  = best_fit_params[2]

                    # append the best fit parameters for MSCN image
                    feat.append(gamma_best)
                    feat.append((lsigma_best*lsigma_best + rsigma_best*rsigma_best)/2)

                    # shifting indices for creating pair-wise products
                    shifts = [[0,1], [1,0], [1,1], [-1,1]] # H V D1 D2

                    for itr_shift in range(1, len(shifts) + 1):
                        OrigArr = structdis
                        reqshift = shifts[itr_shift-1] # shifting index

                        # create transformation matrix for warpAffine function
                        M = np.float32([[1, 0, reqshift[1]], [0, 1, reqshift[0]]])
                        ShiftArr = cv2.warpAffine(OrigArr, M, (structdis.shape[1], structdis.shape[0]))

                        Shifted_new_structdis = ShiftArr
                        Shifted_new_structdis = Shifted_new_structdis * structdis
                        # shifted_new_structdis is the pairwise product
                        # best fit the pairwise product
                        best_fit_params = AGGDfit(Shifted_new_structdis)
                        lsigma_best = best_fit_params[0]
                        rsigma_best = best_fit_params[1]
                        gamma_best  = best_fit_params[2]

                        constant = m.pow(tgamma(1/gamma_best), 0.5)/m.pow(tgamma(3/gamma_best), 0.5)
                        meanparam = (rsigma_best - lsigma_best) * (tgamma(2/gamma_best)/tgamma(1/gamma_best)) * constant

                        # append the best fit calculated parameters
                        feat.append(gamma_best) # gamma best
                        feat.append(meanparam) # mean shape
                        feat.append(m.pow(lsigma_best, 2)) # left variance square
                        feat.append(m.pow(rsigma_best, 2)) # right variance square

                    # resize the image on next iteration
                    im_original = cv2.resize(im_original, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
                return feat

            # read image from given path
            dis = cv2.imread(imgPath, 1)
            if(dis is None):
                print("Wrong image path given")
                print("Exiting...")
                sys.exit(0)
            # convert to gray scale
            dis = cv2.cvtColor(dis, cv2.COLOR_BGR2GRAY)

            # compute feature vectors of the image
            features = compute_features(dis)

            # rescale the brisqueFeatures vector from -1 to 1
            x = [0]

            # pre loaded lists from C++ Module to rescale brisquefeatures vector to [-1, 1]
            min_= [0.336999 ,0.019667 ,0.230000 ,-0.125959 ,0.000167 ,0.000616 ,0.231000 ,-0.125873 ,0.000165 ,0.000600 ,0.241000 ,-0.128814 ,0.000179 ,0.000386 ,0.243000 ,-0.133080 ,0.000182 ,0.000421 ,0.436998 ,0.016929 ,0.247000 ,-0.200231 ,0.000104 ,0.000834 ,0.257000 ,-0.200017 ,0.000112 ,0.000876 ,0.257000 ,-0.155072 ,0.000112 ,0.000356 ,0.258000 ,-0.154374 ,0.000117 ,0.000351]

            max_= [9.999411, 0.807472, 1.644021, 0.202917, 0.712384, 0.468672, 1.644021, 0.169548, 0.713132, 0.467896, 1.553016, 0.101368, 0.687324, 0.533087, 1.554016, 0.101000, 0.689177, 0.533133, 3.639918, 0.800955, 1.096995, 0.175286, 0.755547, 0.399270, 1.095995, 0.155928, 0.751488, 0.402398, 1.041992, 0.093209, 0.623516, 0.532925, 1.042992, 0.093714, 0.621958, 0.534484]

            # append the rescaled vector to x
            for i in range(0, 36):
                min = min_[i]
                max = max_[i]
                x.append(-1 + (2.0/(max - min) * (features[i] - min)))

            # load model
            model = svmutil.svm_load_model("allmodel")

            # create svm node array from python list
            x, idx = gen_svm_nodearray(x[1:], isKernel=(model.param.kernel_type == PRECOMPUTED))
            x[36].index = -1 # set last index to -1 to indicate the end.

            # get important parameters from model
            svm_type = model.get_svm_type()
            is_prob_model = model.is_probability_model()
            nr_class = model.get_nr_class()

            if svm_type in (ONE_CLASS, EPSILON_SVR, NU_SVC):
                # here svm_type is EPSILON_SVR as it's regression problem
                nr_classifier = 1
            dec_values = (c_double * nr_classifier)()

            # calculate the quality score of the image using the model and svm_node_array
            qualityscore = svmutil.libsvm.svm_predict_probability(model, x, dec_values)

            return qualityscore

        qualityscore = test_measure_BRISQUE(img)

        return qualityscore

    def clear_pod_or_not_pre_print(self, img_path, pod_words_pre_print, fifty_percent_of_total_pre_print_words):

        ocr_text = detect_text(img_path)
        ocr_text = [words.lower() for words in ocr_text]

        new_ocr_text = clean_ocr_text(ocr_text[1:])

        ocr_text[1:] = new_ocr_text

        for pod_pp_wrd in pod_words_pre_print:

            for i, ocr_wrd in enumerate(ocr_text[1:]):

                no_of_mistakes = nltk.edit_distance(pod_pp_wrd, ocr_wrd)

                if no_of_mistakes == 1:
                    ocr_text[i] = pod_pp_wrd

                    print('ocr: {}\n pod word: {}'.format(ocr_wrd, pod_pp_wrd))


        pre_print_words_detected = list(set(pod_words_pre_print).intersection(ocr_text[1:]))

        if  len(pre_print_words_detected) >= fifty_percent_of_total_pre_print_words:

            result = 'Clear POD'
            #print(result)

        else:
            result = 'Unclear POD'
            #print(result)

        return result, len(pre_print_words_detected)

    #Clear printed POD or not
    def clear_pod_or_not_printed(self, img_path, pod_words_printed, fifty_percent_of_total_printed_words):

        ocr_text = detect_text(img_path)
        ocr_text = [words.lower() for words in ocr_text]

        new_ocr_text = clean_ocr_text(ocr_text[1:])

        ocr_text[1:] = new_ocr_text

        for pod_printed_wrd in pod_words_printed:

            for i, ocr_wrd in enumerate(ocr_text[1:]):

                no_of_mistakes = nltk.edit_distance(pod_printed_wrd, ocr_wrd)

                if no_of_mistakes == 1:
                    ocr_text[i] = pod_printed_wrd

                    print('pod word: {}\n ocr: {}'.format(ocr_wrd, pod_printed_wrd))

        printed_words_detected = list(set(pod_words_printed).intersection(ocr_text[1:]))

        if  len(printed_words_detected) >= fifty_percent_of_total_printed_words:

            result = 'Clear POD'
            #print(result)

        else:
            result = 'Unclear POD'
            #print(result)

        return result, len(printed_words_detected)



# class get_invoice_no_class:
#
#     def invoice_no_detector(self, img_path, ocr_text):
#
#         # ocr_text = detect_text(img_path)
#         ocr_text = [words.lower() for words in ocr_text]
#
#         new_ocr_text = clean_ocr_text(ocr_text[1:])
#
#         ocr_text[1:] = new_ocr_text
#
#         invoice_nos = []
#
#         for i, wrd in enumerate(ocr_text[1:]):
#
#             no_of_mistakes = nltk.edit_distance('invoice', wrd)
#
#             if no_of_mistakes == 1:
#
#                 wrd = 'invoice'
#
#                 print("ocr_text[i+1]", ocr_text[i+1])
#
#             # if wrd == 'invoice':
#             #
#             #     print('ocr_text[i+1]', ocr_text[i+2])
#
#             if wrd == 'invoice' and (ocr_text[i+2] == 'no' or ocr_text[i+2] == 'no.'):#Remarks
#                 #print('found invoice no.')
#                 for j in range(i, i+25):
#
#                     if ocr_text[j].isnumeric() and len(ocr_text[j]) >= 3:
#
#                         # print('word: {}'.format(ocr_text[j]))
#                         invoice_nos.append(ocr_text[j])
#
#                     elif ocr_text[j].isalnum() and not ocr_text[j].isalpha() and not ocr_text[j].isdigit() and len(ocr_text[j]) >= 4:
#                         # print('word: {}'.format(ocr_text[j]))
#                         invoice_nos.append(ocr_text[j])
#
#                     # else:
#                     #     print('word: {}'.format('something wrong in invoice no'))
#                     #     invoice_nos.append('something wrong in invoice no')
#
#
#                 break
#
#             # else:
#             #
#             #     print('{}'.format('invoice no not found'))
#             #     invoice_nos.append('invoice no not found')
#
#
#         invoice_nos.sort(key=len, reverse=True)
#
#         print('invoice_nos'.format(invoice_nos))
#
#         return invoice_nos

class get_invoice_no_class:

    def invoice_no_detector(self, img_path, ocr_text):

        # ocr_text = detect_text(img_path)
        ocr_text = [words.lower() for words in ocr_text]

        new_ocr_text = clean_ocr_text(ocr_text[1:])

        ocr_text[1:] = new_ocr_text

        invoice_nos = []

        for i, wrd in enumerate(ocr_text[1:]):

            no_of_mistakes = nltk.edit_distance('invoice', wrd)

            if no_of_mistakes == 1:

                wrd = 'invoice'

                print("ocr_text[i+1]", ocr_text[i+1])

            # if wrd == 'invoice':
            #
            #     print('ocr_text[i+1]', ocr_text[i+2])

            if wrd == 'invoice' and (ocr_text[i+2] == 'no' or ocr_text[i+2] == 'no.'):#Remarks
                #print('found invoice no.')

                ocr_wc = len(ocr_text) - 1

                if ocr_wc - i >= 25:


                    for j in range(i, i+25):

                        if ocr_text[j].isnumeric() and len(ocr_text[j]) >= 4:

                            # print('word: {}'.format(ocr_text[j]))
                            invoice_nos.append(ocr_text[j])

                            print('invoice no1: {}'.format(ocr_text[j]))

                        elif ocr_text[j].isalnum() and not ocr_text[j].isalpha() and not ocr_text[j].isdigit() and len(ocr_text[j]) >= 4:
                            # print('word: {}'.format(ocr_text[j]))
                            invoice_nos.append(ocr_text[j])

                            print('invoice no2: {}'.format(ocr_text[j]))

                        # else:
                        #     print('word: {}'.format('something wrong in invoice no'))
                        #     invoice_nos.append('something wrong in invoice no')


                    break

                else:

                    for j in range(i, ocr_wc):

                        if ocr_text[j].isnumeric() and len(ocr_text[j]) >= 4:

                            # print('word: {}'.format(ocr_text[j]))
                            invoice_nos.append(ocr_text[j])

                            print('invoice no1: {}'.format(ocr_text[j]))

                        elif ocr_text[j].isalnum() and not ocr_text[j].isalpha() and not ocr_text[j].isdigit() and len(ocr_text[j]) >= 4:
                            # print('word: {}'.format(ocr_text[j]))
                            invoice_nos.append(ocr_text[j])

                            print('invoice no2: {}'.format(ocr_text[j]))

                        # else:
                        #     print('word: {}'.format('something wrong in invoice no'))
                        #     invoice_nos.append('something wrong in invoice no')


                    break

            # else:
            #
            #     print('{}'.format('invoice no not found'))
            #     invoice_nos.append('invoice no not found')
        invoice_nos.sort(key=len, reverse=True)

        print('Invoice nos class: ', invoice_nos)
        print('invoice type: ', type(invoice_nos))
        print('invoice length: ', len(invoice_nos))


        return invoice_nos


class stamp_detection_class:

    def detect_angle(self, image):

        image = cv2.imread(str(image), cv2.IMREAD_COLOR)
        mask = np.zeros(image.shape, dtype=np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        adaptive = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,15,4)

        cnts = cv2.findContours(adaptive, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for c in cnts:
            area = cv2.contourArea(c)
            if area < 45000 and area > 20:
                cv2.drawContours(mask, [c], -1, (255,255,255), -1)

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        h, w = mask.shape

        font = cv2.FONT_HERSHEY_SIMPLEX

        org = (0, 185)

        # fontScale
        fontScale = 1

        # Red color in BGR
        color = (0, 0, 255)

        # Line thickness of 2 px
        thickness = 2



        # Horizontal
        if w > h:
            left = mask[0:h, 0:0+w//2]
            right = mask[0:h, w//2:]
            left_pixels = cv2.countNonZero(left)
            right_pixels = cv2.countNonZero(right)

            if left_pixels >= right_pixels:
                angle = 0
            else:
                angle = 180

            text = 'angle: ' + str(angle)

            # Using cv2.putText() method
            image = cv2.putText(image, text, org, font, fontScale,
                             color, thickness, cv2.LINE_AA, False)

    #         cv2.imshow('image', image)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()

            return angle#0 if left_pixels >= right_pixels else 180
        # Vertical
        else:
            top = mask[0:h//2, 0:w]
            bottom = mask[h//2:, 0:w]
            top_pixels = cv2.countNonZero(top)
            bottom_pixels = cv2.countNonZero(bottom)

            if bottom_pixels >= top_pixels:
                angle = 90
            else:
                angle = 270

            text = 'angle: ' + str(angle)

            # Using cv2.putText() method
            image = cv2.putText(image, text, org, font, fontScale,
                             color, thickness, cv2.LINE_AA, False)

    #         cv2.imshow('image', image)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()

            return angle#90 if bottom_pixels >= top_pixels else 270



    # if __name__ == '__main__':
    #     image = cv2.imread('1.png')
    #     angle = detect_angle(image)
    #     print(angle)


    def non_stamp_regions(self, img_name):

        img = cv2.imread(img_name)

    #     print('img shape: {}'.format(img.shape))

        angle = self.detect_angle(img_name)

        # font = cv2.FONT_HERSHEY_SIMPLEX
        #
        # org = (0, 185)
        #
        # # fontScale
        # fontScale = 1
        #
        # # Red color in BGR
        # color = (0, 0, 255)
        #
        # # Line thickness of 2 px
        # thickness = 2
        #
        # text = 'angle: ' + str(angle)

        if angle == 0 or angle == 180:

            # print('Angle 0 or 180: ', angle)

            y = 0
            h = int(img.shape[0]/2)
            x = 0
            w = int(img.shape[1])

            # print('y: {}, h: {}, x: {}, w: {}'.format(y, h, x, w))

            # img = cv2.putText(img, text, org, font, fontScale,
            #              color, thickness, cv2.LINE_AA, False)

            crop_img = img[y:y+h, x:x+w]
    #         cv2.imshow("img", img)
    #         cv2.imshow("cropped", crop_img)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()

            return y, y+h, x, x+w#crop_img

        elif angle == 90:

            # print('Angle 90: ', angle)

            y = 0
            h = int(img.shape[0]/2)
            x = 0
            w = int(img.shape[1])

            # print('y: {}, h: {}, x: {}, w: {}'.format(y, h, x, w))

            # img = cv2.putText(img, text, org, font, fontScale,
            #              color, thickness, cv2.LINE_AA, False)


            crop_img = img[y:y+h, x+w:x]
    #         cv2.imshow("img", img)
    #         cv2.imshow("cropped", crop_img)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()

            return y, y+h, x+w, x#crop_img


        elif angle == 270:

            # print('Angle 270: ', angle)

            y = 0
            h = int(img.shape[0])
            x = int(img.shape[1]/2)
            w = img.shape[1]

            crop_img = img[y:y+h, x:w]

            # print('y: {}, h: {}, x: {}, w: {}'.format(y, h, x, w))

            # img = cv2.putText(img, text, org, font, fontScale,
            #              color, thickness, cv2.LINE_AA, False)

    #         cv2.imshow("img", img)
    #         cv2.imshow("cropped", crop_img)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()

            return y, y+h, x, w#crop_img



    def blue_pixels_recog(self, img_name):

        img = cv2.imread(img_name)

    #     cv2.imshow('original image', img)
        #print(img.shape)


        a, b, c, d = self.non_stamp_regions(img_name)

        img[a:b, c:d] = 0

        #cv2.imshow('masked image', img)



        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #     lower_range = np.array([120,50,50])#[110,50,50]100
    #     upper_range = np.array([170,255,255])#[130,255,255]150

        lower_range = np.array([100,50,50])
        upper_range = np.array([130,255,255])

        # Red color
    #     lower_range = np.array([161, 155, 84])
    #     upper_range = np.array([179, 255, 255])

    #     font = cv2.FONT_HERSHEY_SIMPLEX

    #     #     org_1 = int((b-a)/2)
    #     #     org_2 = int((d-c)/2)

    #     org = (int(img.shape[0]), int(img.shape[1]/2 + 20))

    #     # fontScale
    #     fontScale = 0.5

    #     # Red color in BGR
    #     color = (100, 255, 100)

    #     # Line thickness of 2 px
    #     thickness = 2

        # font = cv2.FONT_HERSHEY_SIMPLEX
        #
        # org = (0, 185)
        #
        # # fontScale
        # fontScale = 1
        #
        # # Red color in BGR
        # #color = (0, 0, 255)
        # color = (255, 50, 50)
        #
        # # Line thickness of 2 px
        # thickness = 2


        mask = cv2.inRange(hsv, lower_range, upper_range)

        # cv2.imshow('only stamp region', img)
    #         cv2.imshow('opening', opening)
    #         cv2.waitKey()


        blue_pixels = mask > 0

        blue_pixels_length = mask[blue_pixels]

        if len(blue_pixels_length) > 500:

            print('no. of blue pixels: {}'.format(len(blue_pixels_length)))

            # text1 = 'Stamp detected'
            # org = (0, 185)
            # image_ = cv2.putText(mask, text1, org, font, fontScale,
            #                      color, thickness, cv2.LINE_AA, False)
            #
            # text2 = 'blue pixels: ' + str(len(blue_pixels_length))
            # org = (0, 230)
            # image_ = cv2.putText(mask, text2, org, font, fontScale,
            #                      color, thickness, cv2.LINE_AA, False)

            # cv2.imshow('Stamp detected', image_)
            #
            # #         cv2.imshow('concatenated', concat_img_new)
            #
            # k = cv2.waitKey(0)
            #
            # #         cv2.imwrite(os.path.join(path , img_name), concat_img_new)
            #
            # cv2.destroyAllWindows()

            # stamp_detected = text1 + '\n' + text2

            blue_pixels = 'blue pixels higher than threshold'

        else:

            print('no. of blue pixels: {}'.format(len(blue_pixels_length)))

            # text1 = 'Stamp not detected'
            # org = (0, 185)
            # image_ = cv2.putText(mask, text1, org, font, fontScale,
            #                      color, thickness, cv2.LINE_AA, False)
            #
            # text2 = 'blue pixels: ' + str(len(blue_pixels_length))
            # org = (0, 230)
            # image_ = cv2.putText(mask, text2, org, font, fontScale,
            #                      color, thickness, cv2.LINE_AA, False)
    #         text2 = 'blue pixels: ' + str(len(blue_pixels_length))
    #         text = text1 + '\n' + text2


    #         image_ = cv2.putText(mask, text, org, font, fontScale,
    #                              color, thickness, cv2.LINE_AA, False)

            # cv2.imshow('Stamp detected', image_)
            #
            # #         cv2.imshow('concatenated', concat_img_new)
            #
            # k = cv2.waitKey(0)
            #
            # #         cv2.imwrite(os.path.join(path , img_name), concat_img_new)
            #
            # cv2.destroyAllWindows()

            # stamp_detected = text1 + '\n' + text2

            blue_pixels = 'blue pixels lesser than threshold'

        print(blue_pixels)

        return blue_pixels


    def get_consignee_add(self, image_path, ocr_text, pod_words_printed_consignee):

        img = cv2.imread(image_path)

    #     print('img: {}'.format(image_path))

        ocr_text = [words.lower() for words in ocr_text]

        new_ocr_text = clean_ocr_text(ocr_text[1:])

        ocr_text[1:] = new_ocr_text

        consignee_address = []
        consignee_found = 0

        for i, wrd in enumerate(ocr_text):

            no_of_mistakes = nltk.edit_distance('consignee:', wrd)

            if no_of_mistakes == 1 and ocr_text[i+1] != 'copy' and len(wrd) >= 3:

                wrd = 'consignee:'

            #print(i, wrd)
            if wrd == 'consignee:':

                consignee_found += 1

    #             print('search from: ', i+3)

                for j in range(i+1, i+6):

                    #print(ocr_text[j])

                    if ocr_text[j] in ocr_text[j+10:] and ocr_text[j].lower() not in pod_words_printed_consignee and len(ocr_text[j]) >= 3 and ocr_text[j].lower() != 'ltd':

                        print('consignee name: {}'.format(ocr_text[j]))

                        consignee_address.append(ocr_text[j])

                    else:

                        for wrd in ocr_text[j+10:]:

                            #print(wrd)

                            no_of_mistakes = nltk.edit_distance(ocr_text[j].lower(), wrd.lower())

                            if no_of_mistakes == 1 and ocr_text[j].lower() not in pod_words_printed_consignee and wrd.lower() not in pod_words_printed_consignee and len(ocr_text[j]) >= 3 and len(wrd) >= 3 and wrd.lower() != 'electrical' and wrd.lower() != 'per' and ocr_text[j].lower() != 'ltd':

                                #wrd = ocr_text[j]
    #                             print('wrd: {}'.format(wrd))
    #                             print('ocr: {}'.format(ocr_text[j]))

                                print('consignee name: {}\nstamp consignee name: {}'.format(ocr_text[j], wrd))

                                consignee_address.append(ocr_text[j])





                break



        if consignee_found == 0:

            print('Consignee not found')
        # else:
        #
        #     print('Consignee found')

        if len(consignee_address) == 0:

            detected = 'Consignee not detected'

            print(detected)
        else:
            print('consignee_address: ', consignee_address)
            detected = 'Consignee detected'
            print(detected)

        # font = cv2.FONT_HERSHEY_SIMPLEX
        #
        # org = (0, 185)
        #
        # # fontScale
        # fontScale = 1
        #
        # # Red color in BGR
        # #color = (0, 0, 255)
        # color = (255, 50, 50)
        #
        # # Line thickness of 2 px
        # thickness = 2
        #
        # img = cv2.putText(img, detected, org, font, fontScale,
        #                  color, thickness, cv2.LINE_AA, False)


        # Show result
        # cv2.imshow('stamp detected', img)
        # cv2.waitKey(0)
        #
        # cv2.destroyAllWindows()


        return detected




    def circle_detector(self, img_name):

        img = cv2.imread(img_name)

        a, b, c, d = self.non_stamp_regions(img_name)

        img[a:b, c:d] = 0

        img_blur = cv2.medianBlur(img, 3)
        # Convert to gray-scale
        gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
        # Blur the image to reduce noise
    #     img_blur = cv2.medianBlur(gray, 3)
        #img_blur = cv2.blur(gray,(3,3))
        #img_blur = cv2.GaussianBlur(gray,(3,3),0)

        # Apply hough transform on the image
    #     circles = cv2.HoughCircles(img_blur, cv2.HOUGH_GRADIENT, 1, img.shape[0]/64, param1=200, param2=20, minRadius=50, maxRadius=60)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 25, param1=120, param2=23, minRadius=50, maxRadius=60)#, param1=200, param2=20, minRadius=50, maxRadius=60
        # print(circles)

        # Draw detected circles

        # font = cv2.FONT_HERSHEY_SIMPLEX
        #
        # org = (0, 185)
        #
        # # fontScale
        # fontScale = 1
        #
        # # Red color in BGR
        # color = (0, 0, 255)
        #
        # # Line thickness of 2 px
        # thickness = 2

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Draw outer circle
                cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
                # Draw inner circle
                cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

            # text = 'Circle detected'
            #
            # img = cv2.putText(img, text, org, font, fontScale,
            #              color, thickness, cv2.LINE_AA, False)


            # Show result
    #         cv2.imshow("Circle detected", img)
    #         cv2.waitKey(0)
    #
    # #         path = 'mahindra_docs/BOT POC data/Clear/Stamp_circle_detector'
    #
    # #         img_name = name.split('/')[-1]
    #
    # #         cv2.imwrite(os.path.join(path , img_name), img)
    #
    #         cv2.destroyAllWindows()

            circle_detected = 'Circle detected'

        else:

            circle_detected = 'Circle not detected'

        print(circle_detected)

        return circle_detected

    def stamp_clear_or_not(self, img_name, ocr_text, pod_words_printed_consignee):

        circle_detected = self.circle_detector(img_name)

        if circle_detected == 'Circle detected':

            stamp_detected = 'stamp detected'

            return stamp_detected

        else:

            # consignee = self.get_consignee_add(img_name, ocr_text, pod_words_printed_consignee)
            #
            # if consignee == 'Consignee detected':
            #
            #     stamp_detected = 'stamp detected'
            #
            #     return stamp_detected
            #
            # else:

            blue_pixels = self.blue_pixels_recog(img_name)

            if blue_pixels == 'blue pixels higher than threshold':

                stamp_detected = 'stamp detected'

                return stamp_detected

            else:

                stamp_detected = 'stamp not detected'

                return stamp_detected
