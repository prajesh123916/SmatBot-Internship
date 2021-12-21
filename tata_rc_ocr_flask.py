import requests
from flask import Flask, request, make_response
from flask import jsonify
import time
import os

from tata_rc_ocr_class import regex_checker
from tata_rc_ocr_class import pre_processing_image
from tata_rc_ocr_class import detect_text_ocr
from tata_rc_ocr_class import rc_detection

from datetime import datetime

import urllib.request

import logging

from helper_function import detect_text

my_logfile = datetime.now().strftime('tata_logfile_%H_%M_%d_%m_%Y.log')

#Create and configure logger
logging.basicConfig(filename=my_logfile,
                    format='%(asctime)s %(message)s',
                    filemode='w')

#Creating an object
logger=logging.getLogger()

#Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/monu/Mahindra_image_processing/feltso-144907-02f55b94eddd.json'
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/ubuntu/TataOCR/production_folder/feltso-144907-02f55b94eddd.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/ubuntu/speech_recognition/speech_ai/mahindra_ocr/feltso-144907-02f55b94eddd.json'
# from helper_function import detect_text

regex_checker = regex_checker()
pre_processing_image = pre_processing_image()
detect_text_ocr = detect_text_ocr()
rc_detection = rc_detection()


application = Flask(__name__)

allowed_extensions = set(['jpg', 'jpeg', 'pdf'])

# upload_folder = '/home/monu/TataOCR/uploads/'
# upload_folder = '/home/ubuntu/TataOCR/uploads/'
upload_folder = '/home/ubuntu/speech_recognition/speech_ai/tata_ocr/Tata_data/uploads/'

def allowed_file(filename):

    #print(filename.split('/')[-1].split('.')[-1].lower())
    #return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions
    return filename.split('/')[-1].split('.')[-1].lower() in allowed_extensions


@application.route('/hello_world', methods=['GET','POST'])

def hello_world():

    return 'Hello World'



@application.route('/detect_rc', methods=['GET','POST'])

def extract_rc_details():

    overall_start = time.time()

    if request.method == 'POST':

        try:

            json_file = request.json

            print(json_file)

            logger.info('json file: {}'.format(json_file))

            filename = json_file['image_url']

            if filename == '':

                print('No File')
                logger.info('No File')

                return 'No file'

            elif allowed_file(filename):

                fn = filename.split('/')[-1]

                file_name = upload_folder + fn

                print('\n\nOriginal file name with time stamp: {}\n\n'.format(file_name))

                logger.info("\n\nOriginal file name with time stamp: {}\n\n".format(file_name.split('/')[-1]))

                #file.save(os.path.join(upload_folder, fn))
                urllib.request.urlretrieve(filename, file_name)

                if file_name.split('.')[-1] == 'jpeg' or file_name.split('.')[-1] == 'jpg':# or name.split('.')[-1] == 'pdf':


    #--------------------------------------------------------------------------------------------------------
            # if 'file' not in request.files:
            #
            #     return 'No file'
            #
            # file = request.files['file']
            #
            # if file.filename == '':
            #
            #     return 'No file'
            #
            # if file and not allowed_file(file.filename):
            #
            #     return 'File is not of image format i.e jpg, jpeg, pdf...'
            #
            # if file and allowed_file(file.filename):
            #
            #     time_stamp = '_' + str(time.time()).split('.')[0]
            #
            #     fn = file.filename
            #
            #     fn = fn.rsplit('.', 1)[0] + time_stamp + '.' + fn.rsplit('.', 1)[1]
            #
            #     file_name = upload_folder + fn
            #
            #     print('\n\nOriginal file name with time stamp: {}\n\n'.format(file_name) )
            #
            #     file.save(os.path.join(upload_folder, fn))
            #
            #     if file_name.split('.')[-1] == 'jpeg' or file_name.split('.')[-1] == 'jpg':# or name.split('.')[-1] == 'pdf':
            #
            #         print('Image: {}'.format(file_name))
    #-------------------------------------------------------------------------------------------------------------------------

                    cropped_img_name = pre_processing_image.crop_id_region(file_name)
                    logger.info('\ncropped_img_name: {}\n'.format(cropped_img_name))

                    skewed_img_name = pre_processing_image.deskew_image(cropped_img_name)

                    # print('skewed_img_name: {}\n'.format(skewed_img_name))
                    logger.info('skewed_img_name: {}\n'.format(skewed_img_name))
                #     print('skewed: ', skewed_img_name)
                    rc_type, angle, cordinate = pre_processing_image.detect_rotation_angle(skewed_img_name)
                    logger.info('RC type: {}\nAngle: {}\ncordinate: {}\n'.format(rc_type, angle, cordinate))
                #     print('angle: ', angle)
                #     print("cordinate: ", cordinate)
                    rotated_img_name, rotation_angle = pre_processing_image.rotate_img(skewed_img_name, angle)
                    logger.info('rotated_img_name: {}\nrotation_angle: {}'.format(rotated_img_name, rotation_angle))

                #     skewed_img_name = deskew_image(rotated_img_name)
                #     print('rot_img_name: ', rotated_img_name)
                #     print('rot_angle: ', rotation_angle)

                #     crop_img = cv2.imread(cropped_img_name)
                #     cv2.imshow('cropped', crop_img)
                #     skewed_img = cv2.imread(skewed_img_name)
                #     cv2.imshow('skewed', skewed_img)
                #     rot_img = cv2.imread(rotated_img_name)
                #     cv2.imshow('rotated', rot_img)

                #     cv2.waitKey(0)
                #     cv2.destroyAllWindows()

                #     print('Destroy image')

                    try:
                        ocr_text, field_values_dict = rc_detection.detect_rc(rotated_img_name)
                        logger.info('field values dict: '.format(field_values_dict))
                    except Exception as e:
                        print(e)
                        ocr_text, field_values_dict = rc_detection.detect_rc(rotated_img_name)
                        logger.info('field values dict: '.format(field_values_dict))



                    # print('ocr_text: ', ocr_text)

                    field_values_dict_key_list = list(field_values_dict.keys())

                    final_dict = {}

                    final_dict['RC_TYPE'] = rc_type

                    for dict_key in field_values_dict_key_list:

                        values_list = field_values_dict[dict_key]

                        joined_values = ' '.join(values_list)

                        if dict_key == 'chassis':

                            chassis_no_regex = regex_checker.extract_chassis_no_regex(joined_values)
                            print('chassis_no_regex: ', chassis_no_regex)

                            if chassis_no_regex:

                                final_dict['CHASSIS_NUMBER'] = chassis_no_regex

                            # elif len(chassis_no_regex) == 0:
                            #     # ' '.join(ocr[0].split())
                            #     chassis_no_regex = regex_checker.extract_chassis_no_regex(ocr_text[0])
                            #     final_dict['CHASSIS_NUMBER'] = chassis_no_regex

                            else:

                                chassis_no_regex = regex_checker.extract_chassis_no_regex(ocr_text[0])

                                if chassis_no_regex:
                                    final_dict['CHASSIS_NUMBER'] = chassis_no_regex

                                else:

                                    final_dict['CHASSIS_NUMBER'] = values_list


                        elif dict_key == 'engine':

                            engine_no_regex = regex_checker.extract_engine_no_regex(joined_values)

                            if engine_no_regex and (final_dict['CHASSIS_NUMBER'] != engine_no_regex):
                                final_dict['ENGINE_NUMBER'] = engine_no_regex

                            # elif len(engine_no_regex) == 0:
                            #     engine_no_regex = regex_checker.extract_engine_no_regex(ocr_text[0])
                            #     final_dict['ENGINE_NUMBER'] = engine_no_regex
                            #
                            # else:
                            #     final_dict['ENGINE_NUMBER'] = values_list

                            else:

                                engine_no_regex = regex_checker.extract_engine_no_regex(ocr_text[0])

                                if engine_no_regex:
                                    final_dict['ENGINE_NUMBER'] = engine_no_regex

                                else:

                                    final_dict['ENGINE_NUMBER'] = values_list

                        elif dict_key == 'date':

                            date_of_regn_regex = regex_checker.extract_date_of_regn_regex(joined_values)

                            if date_of_regn_regex:
                                final_dict['DATE_OF_REGISTRATION'] = date_of_regn_regex

                            # elif len(date_of_regn_regex) == 0:
                            #     date_of_regn_regex = regex_checker.extract_date_of_regn_regex(ocr_text[0])
                            #     final_dict['DATE_OF _REGISTRATION'] = date_of_regn_regex
                            #
                            # else:
                            #     final_dict['DATE_OF _REGISTRATION'] = values_list


                            else:

                                date_of_regn_regex = regex_checker.extract_date_of_regn_regex(ocr_text[0])

                                if date_of_regn_regex:
                                    final_dict['DATE_OF _REGISTRATION'] = date_of_regn_regex

                                else:

                                    final_dict['DATE_OF _REGISTRATION'] = values_list


                        elif dict_key == 'regn.':

                            regn_no_regex = regex_checker.extract_regn_no_regex(joined_values)

                            if regn_no_regex:
                                final_dict["REGISTRATION_NUMBER"] = [regn_no_regex[0].upper()]

                            # elif len(regn_no_regex) == 0:
                            #     regn_no_regex = regex_checker.extract_regn_no_regex(ocr_text[0])
                            #     final_dict["REGISTRATION_NUMBER"] = regn_no_regex
                            #
                            # else:
                            #     final_dict["REGISTRATION_NUMBER"] = values_list

                            else:

                                regn_no_regex = regex_checker.extract_regn_no_regex(ocr_text[0])

                                if regn_no_regex:
                                    final_dict["REGISTRATION_NUMBER"] = regn_no_regex

                                else:

                                    final_dict["REGISTRATION_NUMBER"] = values_list




                        elif dict_key == 'mth.':

                            mth_yr_of_mfg_regex = regex_checker.extract_mth_yr_of_mfg_regex(joined_values)

                            if mth_yr_of_mfg_regex:

                                final_dict['MONTH_YR_OF_MFG'] = mth_yr_of_mfg_regex

                            # elif len(mth_yr_of_mfg_regex) == 0:
                            #
                            #     mth_yr_of_mfg_regex = regex_checker.extract_mth_yr_of_mfg_regex(ocr_text[0])
                            #
                            #     final_dict['MONTH_YR_OF_MFG'] = mth_yr_of_mfg_regex
                            #
                            # else:
                            #     final_dict['MONTH_YR_OF_MFG'] = values_list

                            else:

                                mth_yr_of_mfg_regex = regex_checker.extract_mth_yr_of_mfg_regex(ocr_text[0])

                                if mth_yr_of_mfg_regex:
                                    final_dict['MONTH_YR_OF_MFG'] = mth_yr_of_mfg_regex

                                else:

                                    final_dict['MONTH_YR_OF_MFG'] = values_list

                        elif dict_key == 'fuel':

                            fuel_type = regex_checker.extract_fuel_type(joined_values)

                            if fuel_type:
                                final_dict['FUEL_USED'] = fuel_type

                            # elif len(fuel_type) == 0:
                            #     fuel_type = regex_checker.extract_fuel_type(ocr_text[0])
                            #
                            #     final_dict['FUEL_USED'] = fuel_type
                            #
                            # else:
                            #     final_dict['FUEL_USED'] = values_list

                            else:

                                fuel_type = regex_checker.extract_fuel_type(ocr_text[0])

                                if fuel_type:
                                    final_dict['FUEL_USED'] = fuel_type

                                else:

                                    final_dict['FUEL_USED'] = values_list

    ############################################################################################################################
                        elif dict_key == 'hypothecated':
                            final_dict['HYPOTHECATED_TO'] = values_list

                        elif dict_key == 'colour':

                            final_dict['COLOUR'] = []

                            for value in values_list:

                                if value.isalpha():
                                    final_dict['COLOUR'].append(value)

                                else:
                                    final_dict['COLOUR'] = values_list


                        elif dict_key == 'color':

                            final_dict['COLOR'] = []

                            for value in values_list:

                                if value.isalpha():
                                    final_dict['COLOR'].append(value)

                                else:
                                    final_dict['COLOUR'] = values_list



                        elif dict_key == 'regd.':
                            final_dict['REGD_OWNER'] = values_list

                        elif dict_key == "maker's":
                            final_dict['MAKERS_CLASS'] = values_list
                        else:

                            final_dict[dict_key] = values_list


    #                 for dict_key in field_values_dict_key_list:
    #
    #                     values_list = field_values_dict[dict_key]
    #
    #                     joined_values = ' '.join(values_list)
    #
    #                     if dict_key == 'chassis':
    #
    #                         chassis_no_regex = regex_checker.extract_chassis_no_regex(joined_values)
    #
    #                         if chassis_no_regex:
    #
    #                             final_dict['CHASSIS_NUMBER'] = chassis_no_regex
    #
    #                         else:
    #                             # ' '.join(ocr[0].split())
    #                             chassis_no_regex = regex_checker.extract_chassis_no_regex(ocr_text[0])
    #                             final_dict['CHASSIS_NUMBER'] = chassis_no_regex
    #
    #
    #                     elif dict_key == 'engine':
    #
    #                         engine_no_regex = regex_checker.extract_engine_no_regex(joined_values)
    #
    #                         if engine_no_regex:
    #                             final_dict['ENGINE_NUMBER'] = engine_no_regex
    #
    #                         else:
    #                             engine_no_regex = regex_checker.extract_engine_no_regex(ocr_text[0])
    #                             final_dict['ENGINE_NUMBER'] = engine_no_regex
    #
    #                     elif dict_key == 'date':
    #
    #                         date_of_regn_regex = regex_checker.extract_date_of_regn_regex(joined_values)
    #
    #                         if date_of_regn_regex:
    #                             final_dict['DATE_OF_REGISTRATION'] = date_of_regn_regex
    #
    #                         else:
    #                             date_of_regn_regex = regex_checker.extract_date_of_regn_regex(ocr_text[0])
    #                             final_dict['DATE_OF _REGISTRATION'] = date_of_regn_regex
    #
    #                     elif dict_key == 'regn.':
    #
    #                         regn_no_regex = regex_checker.extract_regn_no_regex(joined_values)
    #
    #                         if regn_no_regex:
    #                             final_dict["REGISTRATION_NUMBER"] = [regn_no_regex[0].upper()]
    #
    #                         else:
    #                             regn_no_regex = regex_checker.extract_regn_no_regex(joined_values)
    #                             final_dict["REGISTRATION_NUMBER"] = regn_no_regex
    #
    #
    #                     elif dict_key == 'mth.':
    #
    #                         mth_yr_of_mfg_regex = regex_checker.extract_mth_yr_of_mfg_regex(joined_values)
    #
    #                         if mth_yr_of_mfg_regex:
    #
    #                             final_dict['MONTH_YR_OF_MFG'] = mth_yr_of_mfg_regex
    #
    #                         else:
    #
    #                             mth_yr_of_mfg_regex = regex_checker.extract_mth_yr_of_mfg_regex(joined_values)
    #
    #                             final_dict['MONTH_YR_OF_MFG'] = mth_yr_of_mfg_regex
    #
    #                     elif dict_key == 'fuel':
    #
    #                         fuel_type = regex_checker.extract_fuel_type(joined_values)
    #
    #                         if fuel_type:
    #                             final_dict['FUEL_USED'] = fuel_type
    #
    #                         else:
    #                             fuel_type = regex_checker.extract_fuel_type(joined_values)
    #
    #                             final_dict['FUEL_USED'] = fuel_type
    #
    # ############################################################################################################################
    #                     elif dict_key == 'hypothecated':
    #                         final_dict['HYPOTHECATED_TO'] = values_list
    #
    #                     elif dict_key == 'colour':
    #
    #                         final_dict['COLOUR'] = []
    #
    #                         for value in values_list:
    #
    #                             if value.isalpha():
    #                                 final_dict['COLOUR'].append(value)
    #
    #
    #                     elif dict_key == 'color':
    #
    #                         final_dict['COLOR'] = []
    #
    #                         for value in values_list:
    #
    #                             if value.isalpha():
    #                                 final_dict['COLOR'].append(value)
    #
    #
    #                     elif dict_key == 'regd.':
    #                         final_dict['REGD_OWNER'] = values_list
    #
    #                     elif dict_key == "maker's":
    #                         final_dict['MAKERS_CLASS'] = values_list
    #                     else:
    #
    #                         final_dict[dict_key] = values_list

                #     detect_rc(rotated_img_name)

                    # print('-'*50 + '\n\n')

                    print(final_dict)

                    logger.info(final_dict)

                    delete_image_files = [file_name, cropped_img_name, skewed_img_name, rotated_img_name]

                    for image_file in delete_image_files:
                        if os.path.exists(image_file):

                            os.remove(image_file)

                            logger.info('Deleting the file: {}'.format(image_file))
                            print('Deleting the file: {}'.format(image_file))

                        else:

                            print("The file does not exist: {}".format(image_file))

                    print('-'*50 + '\n\n')
                    
                    return final_dict

        except Exception as e:

            print('Exception occured: {}'.format(e))

            exception = 'Exception occured: \n\n' + e

            logger.info(exception)

            return exception


if(__name__ == "__main__"):

    application.run(host = '0.0.0.0', port='3636')
