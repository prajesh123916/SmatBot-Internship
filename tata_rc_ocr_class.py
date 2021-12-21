import cv2
import numpy as np
from PIL import Image
import glob

import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import rotate

from deskew import determine_skew
import re

from helper_function import detect_text

# from tata_rc_ocr_flask import #logger

import os

class regex_checker:

    def extract_chassis_no_regex(self, ocr_text):

        if len(ocr_text.split()) >= 10:

            ocr_text_list = ocr_text.replace('\n', ' ').lower().strip().split()

        else:

            ocr_text_list = ocr_text.replace('\n', ' ').lower().strip().split()

        chassis_no = []

        for word in ocr_text_list:

        #         print(word)

            regex = re.findall('^(?=.*[a-zA-Z])*(?=.*[0-9])[a-zA-Z0-9*~"-]*$',word)

            #regex = re.findall('^(?=.*[a-zA-Z])\*(?=.*[0-9])[A-Za-z0-9*~]+$', word)# r'(?:\d+[a-zA-Z]+|[a-zA-Z]+\d+)'

            if regex and (regex[0].startswith('m') or regex[0].startswith('w')):

        #             print(regex)
        #             print(regex[0])

                chassis_no.append(regex[0].upper())

            elif len(ocr_text.split()) < 10 and word.isnumeric():

                chassis_no.append(word.upper())

        print('Chassis no: {}'.format(chassis_no))

        #logger.info('Chassis no: {}'.format(chassis_no))


        return chassis_no

    def extract_engine_no_regex(self, ocr_text):

        if len(ocr_text.split()) >= 10:

            ocr_text_list = ocr_text.replace('\n', ' ').lower().strip().split()

        else:

            ocr_text_list = ocr_text.replace('\n', ' ').lower().strip().split()

        engine_no = []

        for word in ocr_text_list:

    #         print(word)

            regex = re.findall('^(?=.*[a-zA-Z])(?=.*[0-9])[A-Za-z0-9.-]+$', word)# r'(?:\d+[a-zA-Z]+|[a-zA-Z]+\d+)'

            if regex and not regex[0].startswith('m'):

                print(regex)

                engine_no.append(regex[0].upper())

            elif len(ocr_text.split()) < 10 and word.isnumeric():

                engine_no.append(word.upper())


        if len(engine_no) > 1:

            no_starts_with_r = []
            no_starts_with_m_w = []
            no_is_numeric = []
            no_without_r_m_w = []

            for no in engine_no:

                if no.startswith('R'):
                    no_starts_with_r.append(no)

                elif no.startswith('M') or no.startswith('W'):
                    no_starts_with_m_w.append(no)

                elif no.isnumeric():
                    no_is_numeric.append(no)

                else:
                    no_without_r_m_w.append(no)

            if len(no_without_r_m_w) > 0:
                engine_no = no_without_r_m_w

            elif len(no_is_numeric) > 0:
                engine_no = no_is_numeric

            elif len(no_starts_with_m_w) > 0:
                engine_no = no_starts_with_m_w

            elif len(no_starts_with_r) > 0:
                engine_no = no_starts_with_r

            else:
                engine_no = engine_no

        else:
            engine_no = engine_no

        print('Engine no: ', engine_no)

        # logger.info('Engine no: ', engine_no)

        return engine_no

    def extract_date_of_regn_regex(self, ocr_text):

        if len(ocr_text.split()) >= 10:

            ocr_text_list = ocr_text.replace('\n', ' ').lower().strip().split()

        else:

            ocr_text_list = ocr_text.replace('\n', ' ').lower().strip().split()

        date_of_regn = []

        dates = []
        years = []
        for word in ocr_text_list:

            print(word)

            regex = re.findall('^([0]?[1-9]|[1|2][0-9]|[3][0|1])[./-]([0]?[1-9]|[1][0-2])[./-]([0-9]{4}|[0-9]{2})$', word)


            if regex:

                dates.append('/'.join(regex[0]))
                years.append(regex[0][-1])

        if len(dates) > 0:

            lower_yr_idx = years.index(min(years))

            print('Date of registration: {}'.format(dates[lower_yr_idx]))

            # logger.info('Date of registration: {}'.format(dates[lower_yr_idx]))

            date_of_regn.append(dates[lower_yr_idx])

            return date_of_regn

        else:

            print('Date of registration not found: {}'.format(dates))

            #logger.info('Date of registration not found: {}'.format(dates))

            return date_of_regn

    def extract_regn_no_regex(self, ocr_text):

        if len(ocr_text.split()) >= 10:

            ocr_text_list = ocr_text.replace('\n', ' ').lower().strip().split()

        else:

            ocr_text_list = ocr_text.replace('\n', ' ').lower().strip().split()

        reg_no = []

        for word in ocr_text_list:

            regex = re.findall('^(?=.*[a-zA-Z])(?=.*[0-9])[A-Za-z0-9]+$', word)

            if regex and (regex[0].startswith('ts') or regex[0].startswith('ap')):

                print(regex)

                reg_no.append(regex[0].upper())

        print('Registration no: {}'.format(reg_no))
        #logger.info('Registration no: {}'.format(reg_no))

        return reg_no

    def extract_mth_yr_of_mfg_regex(self, ocr_text):

        if len(ocr_text.split()) >= 10:

            ocr_text_list = ocr_text.replace('\n', ' ').lower().strip().split()

        else:

            ocr_text_list = ocr_text.replace('\n', ' ').lower().strip().split()

        mth_yr_of_mfg = []

        for word in ocr_text_list:

    #         print(word)

            regex = re.findall('^([0]?[1-9]|[1][0-2])[./-]([0-9]{4})$', word)#|[0-9]{2}


            if regex:

                print(regex)

                mth_yr_of_mfg.append(regex[0][0].upper())
                mth_yr_of_mfg.append(regex[0][1].upper())

        print('Month & Yr. of mfg: {}'.format(mth_yr_of_mfg))

        #logger.info('Month & Yr. of mfg: {}'.format(mth_yr_of_mfg))

        return mth_yr_of_mfg

    def extract_fuel_type(self, ocr_text):

        fuel_type = ['petrol', 'petrol lpg', 'diesel']

        fuel_detected = []

        if len(ocr_text.split()) >= 10:

            ocr_text_list = ocr_text.replace('\n', ' ').lower().strip().split()

        else:

            ocr_text_list = ocr_text.replace('\n', ' ').lower().strip().split()

        for fuel in fuel_type:

            if fuel in ocr_text_list:



                fuel_detected.append(fuel.upper())

        print('Fuel type: {}'.format(fuel_detected))

        #logger.info('Fuel type: {}'.format(fuel_detected))

        return fuel_detected

class pre_processing_image:

    def crop_id_region(self, input_image):

        img = cv2.imread(input_image) # Read in the image and convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = 255*(gray < 128).astype(np.uint8) # To invert the text to0 white
        coords = cv2.findNonZero(gray) # Find all non-zero points (text)
        x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
        rect = img[y:y+h, x:x+w] # Crop the image - note we do this on the original image
    #     cv2.imshow("original", img)
    #     cv2.imshow("Cropped", rect) # Show it
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

        split_img_ext_name = input_image.split('.')[-1]
        #split_img_name = input_image.rsplit('.', 1)[:-1][0]
        split_img_name = input_image.split('.' + split_img_ext_name)[0].split('/')[-1]
    #     print(split_img_name)
    #     print(split_img_ext_name)

        # folder_name = './Tata_data/cropped_images/'
        folder_name = '../Tata_data/cropped_images/'

        cropped_img_name = folder_name + split_img_name + '_cropped.' + split_img_ext_name

        print('\ncropped_img_name: {}\n'.format(cropped_img_name))
        #logger.info('\ncropped_img_name: {}\n'.format(cropped_img_name))

        cv2.imwrite(cropped_img_name, rect) # Save the image

        return cropped_img_name

    def detect_rotation_angle(self, skewed_img):

        im  = cv2.imread(skewed_img)
        h, w, c = im.shape

    #     cv2.imshow('cordinate_no', im)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

        ocr_text, text_bounding_box = detect_text(skewed_img)
        # (cond1 AND/OR COND2) AND/OR (cond3 AND/OR cond4)

        to_detect_back = ['chassis', 'engine', 'date', 'hypothecated', 'colour', 'color', 'cubic', 'wheel', 'seating', 'unladen', 'tax',  'regn.',]#

        to_detect_front = ['regn.', 'regd.', "maker's", 'mth.', 'fuel', 'address', 'vehicle', 'type']#


        # if ('chassis' in text_bounding_box.keys() and 'telangana' in text_bounding_box.keys()) or ('cubic' in text_bounding_box.keys() and 'telangana' in text_bounding_box.keys()) or ('address' in text_bounding_box.keys() and (('colour' in text_bounding_box.keys() or 'color' in text_bounding_box.keys()))):
        if any(to_detect_back) in text_bounding_box.keys() and any(to_detect_front) in text_bounding_box.keys():
            x_ch1 = 0
            y_ch1 = 0

            print('Both front and back copy\n')
            #logger.info('Both front and back copy\n')

            rc_type = 'FRONT_BACK'

        elif 'chassis' in text_bounding_box.keys():

            x_ch1 = text_bounding_box['chassis'][0][0]
            y_ch1 = text_bounding_box['chassis'][1][0]

            print('Its a Back copy\n')
            #logger.info('Its a Back copy\n')

            rc_type = 'BACK'

        elif 'cubic' in text_bounding_box.keys():

            x_ch1 = text_bounding_box['cubic'][0][0]
            y_ch1 = text_bounding_box['cubic'][1][0]
            print('Its a Back copy\n')

            #logger.info('Its a Back copy\n')

            rc_type = 'BACK'

        elif 'date' in text_bounding_box.keys():

            x_ch1 = text_bounding_box['date'][0][0]
            y_ch1 = text_bounding_box['date'][1][0]
            print('Its a Back copy\n')

            #logger.info('Its a Back copy\n')

            rc_type = 'BACK'

        elif 'telangana' in text_bounding_box.keys():

            x_ch1 = text_bounding_box['telangana'][0][0]
            y_ch1 = text_bounding_box['telangana'][1][0]

            print('Its a Front copy\n')

            #logger.info('Its a Front copy\n')

            rc_type = 'FRONT'

        elif 'andhra' in text_bounding_box.keys():

            x_ch1 = text_bounding_box['andhra'][0][0]
            y_ch1 = text_bounding_box['andhra'][1][0]

            print('Its a Front copy\n')
            #logger.info('Its a Front copy\n')

            rc_type = 'FRONT'

        elif 'address' in text_bounding_box.keys():

            x_ch1 = text_bounding_box['address'][0][0]
            y_ch1 = text_bounding_box['address'][1][0]

            print('Its a Front copy\n')
            #logger.info('Its a Front copy\n')

            rc_type = 'FRONT'

        else:

            x_ch1 = 0
            y_ch1 = 0

            print('keywords not found\n')
            #logger.info('keywords not found\n')

            rc_type = 'UNKNOWN'


        print('w: {}'.format(w))
        print('x_ch1: {}'.format(x_ch1))
        print('h: {}'.format(h))
        print('y_ch1: {}\n'.format(y_ch1))

        if (x_ch1 == 0) and (y_ch1 == 0):

            cordinate = 'Both front and back so no cordinate'
            angle = 0

        elif (x_ch1 < w/2) and (y_ch1 < w/2):

            cordinate = '1st cordinate'
            angle = 0

    #         print('1st cordinate rotation angle ', angle)

        elif (x_ch1 > w/2) and (y_ch1 < h/2):

            cordinate = '2nd cordinate'
            angle = 270

    #         print('2nd cordinate and rotation angle', angle)

        elif (x_ch1 > w/2) and (y_ch1 > h/2):

            cordinate = '3rd cordinate'
            angle = 180

    #         print('3rd cordinate and rotation angle', angle)

        elif (x_ch1 < w/2) and (y_ch1 > h/2):

            cordinate = '4th cordinate'
            angle = 90

            # print('4th cordinate and rotation angle', angle)

        else:

            cordinate = 'Cordinate unknown'
            angle = 0

            # print('Not able to find the cordinate and angle')

    #     print('Show image')


    #     cv2.imshow('cordinate_no', im)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    #     print('Destroy image')

        # try:
        #     print('Angle: {}'.format(angle))
        #     print('Coordinate: {}\n'.format(cordinate))
        #
        # except Exception as e:
        #     print('Not able to find the cordinate')


        return rc_type, angle, cordinate


    def deskew_image(self, input_image):

        image = io.imread(input_image)


        grayscale = rgb2gray(image)
        angle = determine_skew(grayscale)
        rotated = rotate(image, angle, resize=True) * 255

    #     split_img_name = input_image.rsplit('.', 1)[:-1][0]
    #     split_img_ext_name = input_image.rsplit('.', 1)[-1]

        split_img_ext_name = input_image.split('.')[-1]
        #split_img_name = input_image.rsplit('.', 1)[:-1][0]
        split_img_name = input_image.split('.' + split_img_ext_name)[0].split('/')[-1]
    #     print(split_img_name)
    #     print(split_img_ext_name)

        print('skewed angle: {}\n'.format(angle))
        #logger.info('skewed angle: {}\n'.format(angle))

        # folder_name = './Tata_data/skewed_images/'
        folder_name = '../Tata_data/skewed_images/'

        skewed_img_name = folder_name + split_img_name + '_skewed.' + split_img_ext_name

    #     print(split_img_name)
    #     #print(split_img_name)
    #     print(skewed_img_name)

        print('skewed_img_name: {}\n'.format(skewed_img_name))
        #logger.info('skewed_img_name: {}\n'.format(skewed_img_name))

        io.imsave(skewed_img_name, rotated.astype(np.uint8))

        return skewed_img_name

    def rotate_img(self, skewed_img_name, angle):

    #     input_image = '"./WhatsApp Image 2021-07-22 at 09.02.47_pil_rotated_image_2.jpeg"'
        original_Image = Image.open(skewed_img_name)

        if angle == 270:
            rotation_angle = 90
        elif angle == 180:
            rotation_angle = 180
        elif angle == 90:
            rotation_angle = 270
        else:
            rotation_angle = 0

        if rotation_angle == 0:
            #rotated_image = original_Image.rotate(rotation_angle, expand = True)
            rotated_img_name = skewed_img_name

            print('Rotation angle: {}'.format(rotation_angle))

    #         print('Rotation angle: {}'.format(rotation_angle))

    #         split_img_name = skewed_img_name.rsplit('.', 1)[:-1][0]
    #         split_img_ext_name = skewed_img_name.rsplit('.', 1)[-1]

    #         return rotated_img_name, rotation_angle

        else:
            print('Rotation angle: {}'.format(rotation_angle))
            #logger.info('Rotation angle: {}'.format(rotation_angle))

            rotated_image = original_Image.rotate(rotation_angle, expand = True)
    #         print('Rotation angle: {}'.format(rotation_angle))

    #         split_img_name = skewed_img_name.rsplit('.', 1)[:-1][0]
    #         split_img_ext_name = skewed_img_name.rsplit('.', 1)[-1]

            split_img_ext_name = skewed_img_name.split('.')[-1]
            #split_img_name = input_image.rsplit('.', 1)[:-1][0]
            split_img_name = skewed_img_name.split('.' + split_img_ext_name)[0].split('/')[-1]
        #     print(split_img_name)
        #     print(split_img_ext_name)

    #         print(split_img_name)
    #         print(split_img_ext_name)

            # folder_name = './Tata_data/rotated_images/'
            folder_name = '../Tata_data/rotated_images/'

            rotated_img_name = folder_name + split_img_name + '_rotated.' + split_img_ext_name

            rotated_image = rotated_image.save(rotated_img_name)


            print('rotated_img_name: {}'.format(rotated_img_name))
            #logger.info('rotated_img_name: {}'.format(rotated_img_name))

        return rotated_img_name, rotation_angle

class detect_text_ocr:

    def detect_text(self, path):
        """Detects text in the file."""
        from google.cloud import vision
        import io
        client = vision.ImageAnnotatorClient()

        with io.open(path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        response = client.text_detection(image=image)
        texts = response.text_annotations
        #print('Texts:')

        text_bounding_box = {}

        ocr_text = []

        for text in texts:
            x, y = [],[]

    #         print('\n"{}"'.format(text.description))

            if len(text.description) >=0:

        #         vertices = (['({},{})'.format(vertex.x, vertex.y)
        #                     for vertex in text.bounding_poly.vertices])

                for vertex in text.bounding_poly.vertices:
                    x.append(vertex.x)
                    y.append(vertex.y)


    #             print('X: {}'.format(x))
    #             print('Y: {}'.format(y))

    #             text_bounding_box[(text.description).lower()] = []
    #             text_bounding_box[(text.description).lower()].append(x)
    #             text_bounding_box[(text.description).lower()].append(y)

                if text.description.lower() not in text_bounding_box.keys():
                    text_bounding_box[(text.description).lower()] = []
                    text_bounding_box[(text.description).lower()].append(x)
                    text_bounding_box[(text.description).lower()].append(y)
                else:
                    text_bounding_box[(text.description).lower()].append(x)
                    text_bounding_box[(text.description).lower()].append(y)

                ocr_text.append(text.description)

        #         print('bounds: {}'.format(','.join(vertices)))

        if response.error.message:
            raise Exception(
                '{}\nFor more info on error messages, check: '
                'https://cloud.google.com/apis/design/errors'.format(
                    response.error.message))

            exception = str(response.error.message) + '\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors'

            return exception
        else:

            return ocr_text, text_bounding_box


class rc_detection:

    def detect_rc(self, inp_rotated_img):

        to_detect_back = ['chassis', 'engine', 'date', 'hypothecated', 'colour', 'color']#  'cubic', 'wheel', 'seating', 'unladen', 'tax',  'regn.',
    #     to_detect_back = ['tax']
        exclude_back = ['chassis', 'number', 'engine', 'cubic', 'capacity', 'wheel', 'base', 'seating', 'unladen', 'weight', 'colour', 'date', 'of', 'registration', 'registration:', 'regn.', 'valid', 'upto', 'tax', 'hypothecated', 'to', ':']

        to_detect_front = ['regn.', 'regd.', "maker's", 'mth.', 'fuel']#'address', 'vehicle', 'type'

        exclude_front = ['regn.', 'number', 'number:', 'regd.', 'rego.', 'owner', 'owner:''address', "maker's", 'class', 'clas' 'class:', 'vehicle', 'mth.', 'yr.', 'of', 'mfg', 'mfg:', 'fuel', 'used', 'type', 'body', 't', 'of']

        exclude_front_back = exclude_back + exclude_front

        to_detect_front_back = to_detect_back + to_detect_front

        try:

            ocr_text, text_bounding_box = detect_text(inp_rotated_img)

        except Exception as e:
            print('Exception: {}'.format(e))

            #logger.info('Exception: {}'.format(e))


            ocr_text = detect_text(file_name)

        # print(ocr_text[0])

        key_list = list(text_bounding_box.keys())
        # print('key list: {}'.format(key_list))

        # value_list = list(text_bounding_box.values())
        # print('value list: {}'.format(value_list))

        if ('chassis' in key_list and 'telangana' in key_list) or ('cubic' in key_list and 'telangana' in key_list):

            to_detect = to_detect_front_back


        elif 'telangana' in key_list or 'certificate' in key_list  or 'pradesh' in key_list:#'andhra'
            to_detect = to_detect_front



        elif 'chassis' in key_list or 'cubic' in key_list or 'date' in key_list:

            to_detect = to_detect_back


        else:

            to_detect = to_detect_front_back
            print('No keywords detected')

            #logger.info('No keywords detected')


        field_values_dict = {}

        for word_to_detect in to_detect:

            word_to_detect  = word_to_detect.lower()

            print('\nWord to detect: {}'.format(word_to_detect))

            #logger.info('\nWord to detect: {}'.format(word_to_detect))



            field_values_dict[word_to_detect] = []

            try:
    #         print(text_bounding_box[word_to_detect])
                # y1 = text_bounding_box[word_to_detect][1][0]
                # y3 = text_bounding_box[word_to_detect][1][2]
                #
                # if to_detect == to_detect_front:
                #     new_y1 = int(y1 - (y3-y1)/1.5)
                # else:
                #     new_y1 = int(y1 - (y3-y1)/1.5)#1.15

    #             print('y1 of the word to detect: {}\n'.format(y1))
    #             print('y3 of the word to detect: {}\n'.format(y3))
    #             print('new_y1 of the word to detect: {}'.format(new_y1))

                field_values_dict[word_to_detect] = []

                for keyword in key_list:

                    if len(field_values_dict[word_to_detect]) == 0:

                         y1 = text_bounding_box[word_to_detect][1][0]
                         y3 = text_bounding_box[word_to_detect][1][2]

                         if to_detect == to_detect_front:
                             new_y1 = int(y1 - (y3-y1)/1.5)
                         else:
                             new_y1 = int(y1 - (y3-y1)/1.5)#1.15

                    else:

                        y1 = text_bounding_box[new_keyword][1][0]
                        y3 = text_bounding_box[new_keyword][1][2]

                        if to_detect == to_detect_front:
                            new_y1 = int(y1 - (y3-y1)/1.5)
                        else:
                            new_y1 = int(y1 - (y3-y1)/1.5)#1.15

                    if keyword != word_to_detect and len(keyword) < 200:

                        list_length = len(text_bounding_box[keyword])

                        word_appears_times = int(list_length/2)
            #             print('no. of same keyword: {}'.format(word_appears_times))

                        if word_appears_times <= 1:

                            keyword_y1 = text_bounding_box[keyword][1][0]
            #                 print('keyword_y1: {}'.format(keyword_y1))

                            if (keyword_y1 >= new_y1) and (keyword_y1 <= y3):

                                print('keyword: {}'.format(keyword))
            #                     print('keyword_y1 of the keyword: {}\n'.format(keyword_y1))

                                if keyword not in exclude_front_back:

                                    field_values_dict[word_to_detect].append(keyword.upper())

                                    new_keyword = keyword

                        else:

                            for i in range(1, word_appears_times):


                                idx = (i*2) - 1
            #                     print('idx: ',idx)

                                keyword_y1 = text_bounding_box[keyword][idx][0]
            #                     print('keyword_y1: {}'.format(keyword_y1))

                                if (keyword_y1 >= new_y1) and (keyword_y1 <= y3):

                                    print('keyword: {}'.format(keyword))
            #                         print('keyword_y1 of the keyword: {}\n'.format(keyword_y1))

                                    if keyword not in exclude_front_back:

                                        field_values_dict[word_to_detect].append(keyword.upper())

                                        new_keyword = keyword

            #     for keyword in key_list:
            #
            #         if keyword != word_to_detect and len(keyword) < 200:
            #
            #             list_length = len(text_bounding_box[keyword])
            #
            #             word_appears_times = int(list_length/2)
            # #             print('no. of same keyword: {}'.format(word_appears_times))
            #
            #             if word_appears_times <= 1:
            #
            #                 keyword_y1 = text_bounding_box[keyword][1][0]
            # #                 print('keyword_y1: {}'.format(keyword_y1))
            #
            #                 if (keyword_y1 >= new_y1) and (keyword_y1 <= y3):
            #
            #                     print('keyword: {}'.format(keyword))
            #                     #logger.info('keyword: {}'.format(keyword))
            #
            # #                     print('keyword_y1 of the keyword: {}\n'.format(keyword_y1))
            #
            #                     if keyword not in exclude_front_back:
            #
            #                         field_values_dict[word_to_detect].append(keyword.upper())
            #
            #             else:
            #
            #                 for i in range(1, word_appears_times):
            #
            #
            #                     idx = (i*2) - 1
            # #                     print('idx: ',idx)
            #
            #                     keyword_y1 = text_bounding_box[keyword][idx][0]
            # #                     print('keyword_y1: {}'.format(keyword_y1))
            #
            #                     if (keyword_y1 >= new_y1) and (keyword_y1 <= y3):
            #
            #                         print('keyword: {}'.format(keyword))
            #                         #logger.info('keyword: {}'.format(keyword))
            #
            # #                         print('keyword_y1 of the keyword: {}\n'.format(keyword_y1))
            #
            #                         if keyword not in exclude_front_back:
            #
            #                             field_values_dict[word_to_detect].append(keyword.upper())

            except Exception as e:

                print(e)

        print(field_values_dict)
        #logger.info(field_values_dict)

        print('---------------------------------\n')

        return ocr_text, field_values_dict
