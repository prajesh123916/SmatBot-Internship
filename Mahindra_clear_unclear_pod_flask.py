from flask import Flask, request, make_response
from flask import jsonify
from pdf2image import convert_from_path
# from pyzbar import pyzbar
# import cv2
import nltk

import json

import os
import time

import urllib.request

from datetime import datetime

import logging

from helper_func import allowed_file
from helper_func import clean_ocr_text
# from helper_func import detect_text
from helper_func import creating_reasons
from helper_func import allowed_extensions

from clear_unclear_classes import barcode_reader_class
from clear_unclear_classes import pod_classifier_class
from clear_unclear_classes import complete_incomplete_class
from clear_unclear_classes import clear_not_clear_class
from clear_unclear_classes import get_invoice_no_class
from clear_unclear_classes import stamp_detection_class

barcode_reader_class = barcode_reader_class()
pod_classifier_class = pod_classifier_class()
complete_incomplete_class = complete_incomplete_class()
clear_not_clear_class = clear_not_clear_class()
get_invoice_no_class = get_invoice_no_class()
stamp_detection_class = stamp_detection_class()

application = Flask(__name__)

my_logfile = datetime.now().strftime('mahindra_logfile_%H_%M_%d_%m_%Y.log')

#Create and configure logger
logging.basicConfig(filename=my_logfile,
                    format='%(asctime)s %(message)s',
                    filemode='w')

#Creating an object
logger=logging.getLogger()

#Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)


allowed_extensions = set(['jpg', 'jpeg', 'pdf'])

upload_folder = '/home/ubuntu/speech_recognition/speech_ai/mahindra_ocr/uploads/'
#upload_folder = '/home/monu/Mahindra_image_processing/uploads/'





#Pre-print POD words
pod_words_pre_print = ['Mahindra', 'Logistics', 'Ltd',\
         'POD', 'COPY', 'Corp', 'Off', '1A', '&', '1B' '4th' 'floor', 'Techniplex', '1', 'Goregaon', 'Mumbai',\
         '400062', 'Maharashtra', 'PAN', 'No', 'AAFCM2530H', 'GSTIN',  '27FAFCM2530H1ZO','CIN', 'U63000MH2007PLC173466',\
         'Vehicle', 'Type', 'Consignor', 'Address', 'Volume',\
         'GOODS', 'CONSIGNMENT', 'NOTE', 'AT', "OWNER'S", 'RISK', 'Consignee', 'Chargeable',\
         'GCN', 'Date', 'BA', 'Code', 'From', 'To', 'Freight', 'Service', 'Mode',\
         'Received', 'the', 'goods', 'for', 'transportation', 'subject', 'to', 'T', 'Cs',\
         'Invoice', 'Particulars', 'of', 'Goods', 'said', 'to', 'contain', 'Pkg', 'Remarks',\
         'Please', 'do', 'not', 'sign', 'or', 'stamp', 'on', 'the', 'barcode',\
         'Declared', 'Value', 'I/We', 'do', 'hereby', 'that', 'above', 'particulars', 'of', 'goos', 'consigned',\
         'by', 'me', 'us', 'have', 'been', 'correctly', 'entered', 'into', 'and', 'the', 'consignment', 'is',\
         'booked', 'with', 'full', 'knowledge', 'of', 'which', 'I', 'We', 'accept',\
         'Signature', 'of', 'Consignor', 'his', 'Agent', 'or', 'Representative', 'Booking', 'Incharge', 'Terms',\
         'and', 'Conditions', 'Goods', 'Consignments', 'Note', 'are', 'mentioned', 'on', 'http://www.mahindralogistics.com',\
         'If', 'you', 'are', 'unable', 'see', 'may', 'contact', 'on', '1-1800-258-6787', 'All', 'disputes', 'subject',\
         'Jurisdiction', 'only',\
         'Proof', 'delivery', 'Time', 'Received', 'by', 'Name', 'Sign', 'Remarks',\
         'Regd', 'Office', 'Towers', 'P.K.', 'Kurne', 'Chowk', 'Worli', '400018']

pod_words_pre_print = [words.lower() for words in pod_words_pre_print]
#pod_words_pre_print

pod_words_pre_print_unique = list(set(pod_words_pre_print))
#pod_words_pre_print_unique

#Invalid words
invalid_words = ['ACKNOWLEDGMENT', 'Cube', 'Reporting', 'Unloading',\
                 'Detention', 'handover', 'Damages', 'Damage']
invalid_words = [words.lower() for words in invalid_words]
#invalid_words

invalid_words_unique = list(set(invalid_words))
#invalid_words_unique

#Printed POD extra words
printed_pod_extra_words = ['Gross', "Driver's", 'NON', 'NEGOTIABLE', 'USER', 'Phone', 'USER:', 'Phone:']#'NON', 'NEGOTIABLE',
printed_pod_extra_words = [words.lower() for words in printed_pod_extra_words]
#printed_pod_extra_words

printed_pod_extra_words_unique = list(set(printed_pod_extra_words))
#printed_pod_extra_words_unique










# #Boundary words for pre-print POD
# boundary_words_pre_print = ['Mahindra', 'Logistics', 'Vehicle', 'Address', 'Consignor', 'Received', 'Invoice', 'Please',\
#                             'Declared', 'I/We', 'correctly', 'which', 'Signature', 'http', 'contact', 'Regd', 'Towers',\
#                            '400018', 'Remarks', 'Time', 'delivery', 'Mode', 'Freight', 'To', 'From', 'Code', 'Date',\
#                            'G.C.N', 'U63000MH2007PLC173466', 'Maharashtra', 'COPY', 'Mahindra', 'Logistics']

#Boundary words for pre-print POD
boundary_words_pre_print = ['Address', 'Volume', 'Invoice', 'Please',\
                            'Declared', 'Booking', 'Regd', 'office', 'Towers',\
                           '400018', 'Time', 'delivery', 'Mode', 'Freight', 'From', 'Code',\
                           'G.C.N', 'U63000MH2007PLC173466', 'Maharashtra', 'POD', 'COPY']

	#Repeated boundary words for pre-print POD
	repeated_boundary_words_pre_print = {'Mahindra': 5, 'Logistics': 4, 'Ltd.': 3, 'Vehicle': 2, 'Consignor': 3, 'Received': 2, 'Signature':2, 'Remarks': 2, 'Date': 2}


boundary_words_pre_print = [words.lower() for words in boundary_words_pre_print]
boundary_words_pre_print = list(set(boundary_words_pre_print))
#boundary_words_pre_print

boundary_words_pre_print = clean_ocr_text(boundary_words_pre_print)
#boundary_words_pre_print

#Calculate 50% of the total pre-print boundary words
fifty_percent_of_total_pre_print_boundary_words = int((len(boundary_words_pre_print)/100)*70)
#print('Total pre-print boundary words: {}\nfifty percent of total pre-print boundary words: {}'.format(len(boundary_words_pre_print), fifty_percent_of_total_pre_print_boundary_words))

# #Boundary words for printed POD
# boundary_words_printed = ['Mahindra', 'Logistics', 'Vehicle', 'Address', 'Consignor', 'Received', 'Invoice', 'Declared', 'I/We',\
#                          'me/us', 'with', 'Signature', '6787', 'Regd', 'Towers', 'Sign', '1-800-258-', 'Remarks', 'Time', 'delivery',\
#                          'barcode', 'Remarks', 'Mode', 'Freight', 'To', 'From', 'Code', 'Date', 'G.C.N', 'L63000MH2007PLC173466', 'U63000MH2007PLC173466'\
#                          '27FAFCM2530H1ZO', 'Maharashtra', 'COPY']

#Boundary words for printed POD
	boundary_words_printed = ['Address', 'Invoice', 'Declared', 'Volume', 'Booking', 'Regd', 'office',\
		                 'Towers', 'Sign', '1-800-258-', 'Time', 'delivery',\
		                 'barcode', 'Mode', 'Freight', 'From', 'Code', 'G.C.N.No.:', 'Maharashtra', 'POD', 'COPY']

#Repeated boundary words for printed POD
repeated_boundary_words_printed = {'Mahindra': 5, 'Logistics': 4, 'Ltd.': 3, 'Vehicle': 2, 'Consignor': 3, 'Received': 2, 'Signature':2, 'Remarks': 2, 'Date': 2}



boundary_words_printed = [words.lower() for words in boundary_words_printed]
boundary_words_printed = list(set(boundary_words_printed))
#boundary_words_printed

boundary_words_printed = clean_ocr_text(boundary_words_printed)
#boundary_words_printed

#Calculate 50% of the total printed boundary words
fifty_percent_of_total_printed_boundary_words = int((len(boundary_words_printed)/100)*70)
#print('Total printed boundary words: {}\nfifty percent of total printed boundary words: {}'.format(len(boundary_words_printed), fifty_percent_of_total_printed_boundary_words))









#Words in the pre-print POD
pod_words_pre_print = ['Mahindra', 'Logistics', 'Ltd',\
         'POD', 'COPY', 'Corp', 'Off', '1A', '&', '1B' '4th' 'floor', 'Techniplex', '1', 'Goregaon', 'Mumbai',\
         '400062', 'Maharashtra', 'PAN', 'No', 'AAFCM2530H', 'GSTIN',  '27FAFCM2530H1ZO','CIN', 'U63000MH2007PLC173466',\
         'Vehicle', 'Type', 'Consignor', 'Address', 'Volume',\
         'GOODS', 'CONSIGNMENT', 'NOTE', 'AT', "OWNER'S", 'RISK', 'Consignee', 'Chargeable',\
         'GCN', 'Date', 'BA', 'Code', 'From', 'To', 'Freight', 'Service', 'Mode',\
         'Received', 'the', 'goods', 'for', 'transportation', 'subject', 'to', 'T', 'Cs',\
         'Invoice', 'Particulars', 'of', 'Goods', 'said', 'to', 'contain', 'Pkg', 'Remarks',\
         'Please', 'do', 'not', 'sign', 'or', 'stamp', 'on', 'the', 'barcode',\
         'Declared', 'Value', 'I/We', 'do', 'hereby', 'that', 'above', 'particulars', 'of', 'goos', 'consigned',\
         'by', 'me', 'us', 'have', 'been', 'correctly', 'entered', 'into', 'and', 'the', 'consignment', 'is',\
         'booked', 'with', 'full', 'knowledge', 'of', 'which', 'I', 'We', 'accept',\
         'Signature', 'of', 'Consignor', 'his', 'Agent', 'or', 'Representative', 'Booking', 'Incharge', 'Terms',\
         'and', 'Conditions', 'Goods', 'Consignments', 'Note', 'are', 'mentioned', 'on', 'http://www.mahindralogistics.com',\
         'If', 'you', 'are', 'unable', 'see', 'may', 'contact', 'on', '1-1800-258-6787', 'All', 'disputes', 'subject',\
         'Jurisdiction', 'only',\
         'Proof', 'delivery', 'Time', 'Received', 'by', 'Name', 'Sign', 'Remarks',\
         'Regd', 'Office', 'Towers', 'P.K.', 'Kurne', 'Chowk', 'Worli', '400018']

#Calculate 50% of the total pre-print words
fifty_percent_of_total_pre_print_words = int((len(pod_words_pre_print)/100)*50)
#print('Total pre-print words: {}\nfifty percent of total pre-print words: {}'.format(len(pod_words_pre_print), fifty_percent_of_total_pre_print_words))

#Words in the printed POD
pod_words_printed = ['Mahindra', 'Logistics', 'Ltd',\
         'Driver', 'POD', 'COPY', 'Corp', 'Off', '1A', '&', '1B' '4th' 'floor', 'Techniplex', '1', 'Goregaon', 'Mumbai',\
         '400062', 'Maharashtra', 'PAN', 'No', 'AAFCM2530H', 'GSTIN',  '27FAFCM2530H1ZO','CIN', 'L63000MH2007PLC173466', 'U63000MH2007PLC173466',\
         'Vehicle', 'Type', 'Consignor', 'Address', 'Volume', 'Gross',\
         'GOODS', 'CONSIGNMENT', 'NOTE', 'Non' , 'Negotiable', 'AT', "OWNER'S", 'RISK', 'Consignee', 'Chargeable',\
         'GCN', 'Date', 'BA', 'Code', 'From', 'To', 'Freight', 'Service', 'Mode',\
         'Received', 'the', 'goods', 'for', 'transportation', 'subject', 'to', 'T', 'Cs',\
         'Invoice', 'Particulars', 'of', 'Goods', 'said', 'to', 'contain', 'Pkg', 'Remarks',\
         'Please', 'do', 'not', 'sign', 'or', 'stamp', 'on', 'the', 'barcode',\
         'Declared', 'Value', 'I/We', 'do', 'hereby', 'certify', 'that', 'above', 'particulars', 'of', 'goods', 'consigned',\
         'by', 'me', 'us', 'have', 'been', 'correctly', 'entered', 'into', 'and', 'the', 'consignment', 'is',\
         'booked', 'with', 'full', 'knowledge', 'of', 'which', 'I', 'We', 'accept',\
         'Signature', 'of', 'Consignor', 'his', 'Agent', 'or', 'Representative', 'Booking', 'Incharge', 'Terms',\
         'and', 'Conditions', 'Goods', 'Consignments', 'Note', 'are', 'mentioned', 'on', 'http://www.mahindralogistics.com',\
         'If', 'you', 'are', 'unable', 'see', 'may', 'contact', 'on', '1-800-258-6787', 'All', 'disputes', 'subject',\
         'Jurisdiction', 'only',\
         'Proof', 'delivery', 'Time', 'Received', 'by', 'Name', 'Sign', 'Remarks',\
         'Regd', 'Office', 'Towers', 'P.K.', 'Kurne', 'Chowk', 'Worli', '400018', "Driver's", 'Name', 'Sign']

pod_words_printed = [words.lower() for words in pod_words_printed]
pod_words_printed = list(set(pod_words_printed))
#print(pod_words_printed)

pod_words_printed = clean_ocr_text(pod_words_printed)
#print(pod_words_printed)

#Calculate 50% of the total printed boundary words # Prajesh Reccomendation: You have to remove the Boundary word 
fifty_percent_of_total_printed_words = int((len(pod_words_printed)/100)*50)
#print('Total printed words: {}\nfifty percent of total printed words: {}'.format(len(pod_words_printed), fifty_percent_of_total_printed_words))



pod_words_printed_consignee = ['Driver', '/', 'POD', 'Copy', 'Mahindra', 'Logistics', 'Ltd.', 'Corp', 'Off:', '1A', '&', '1B,', '4th', 'Floor,', 'Techniplex', '1,', 'Goregaon',
                    '(W).,', 'Mumbai', '-', '400062,', 'Maharashtra', 'PAN', 'No.:AAFCM2530H', 'GSTIN:27AAFCM2530H120', 'CIN', 'No.:U63000MH2007PLC173466', 'Mahindra',
                    'LOGISTICS', 'Vehicle', 'No.:', 'GOODS', 'CONSIGNMENT', 'NOTE', 'AT', "OWNER'S", 'RISK', 'G.C.N.No.:', '104484370', 'Vehlcle', 'Type.:', 'Date.:',
                     'Consignor:', 'Address:', 'User:', 'Consignee:', 'BA', 'code', 'From', 'To', 'Phone:', 'Address:', 'Consignor', 'Gross', 'Wt', ':', 'Freight:',
                     'Service', 'Mode:', 'Volume:', 'Chargeable', 'Wt', ':', 'Received', 'the', 'goods', 'for', 'transportation', 'subject', 'to', 'the', 'T&Cs:',
                     'Particulars', 'of', 'Goods', '(said', 'to', 'contain)', 'Invoice', 'No', 'No', 'of', 'Pkg', 'Remarks', '*PLEASE', 'DO', 'NOT', 'SIGNOR', 'STAMP',
                     'ON', 'THE', 'BARCODE', 'Declared', 'Value:', 'IWe', 'do', 'hereby', 'certify', 'that', 'the', 'above', 'particulars', 'of', 'goods', 'consigned',
                     'by', 'me/us', 'have', 'been', 'correctly', 'entered', 'into', 'and', 'the', 'consignement', 'is', 'booked', 'with', 'full', 'knowledge', 'of',
                     'the', 'T&Cs,', 'which', 'IWe', 'I/We' 'accept.', 'Proof', 'of', 'delivery', 'Received', 'by', '(Name', 'Sign):', 'Date.:', 'Time.:', 'Remarks',
                     'Signature', 'of', 'Consignor,', 'his', 'Agent', 'or', 'Representative', 'Signature', 'of', 'Booking', 'Incharge', 'for', 'Mahindra', 'Logistics',
                     'Ltd.', '(Network)', 'The', 'Terms', 'and', 'Conditions', 'of', 'this', 'Goods', 'Consignments', 'Note', 'are', 'mentioned', 'on', 'http://www.mahindralogistics.com',
                     "['T&Cs'],", 'If', 'you', 'are', 'unable', 'to', 'see', 'the', 'T&Cs', 'you', 'may', 'contact', 'on', '1-800-258-', '6787.', 'All', 'disputes', 'are',
                     'subject', 'to', 'Mumbai', 'Jurisdiction', 'only.' 'Regd.,', 'office:', 'Mahindra', 'Logistics', 'Ltd,', 'Mahindra', 'Towers,', 'P.K.Kume', 'Chowk,',
                     'Worli,', 'Mumbai-400018.', "Driver's", 'Name', '&', 'Sign:'
                     ]

pod_words_printed_consignee = [wrd.lower() for wrd in pod_words_printed_consignee]

#Setting up the environment credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/ubuntu/speech_recognition/speech_ai/mahindra_ocr/feltso-144907-02f55b94eddd.json'
#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/monu/Mahindra_image_processing/feltso-144907-02f55b94eddd.json'

from google.cloud import vision
client = vision.ImageAnnotatorClient()



#OCR function using GCP Vision API
def detect_text(path):
    """Detects text in the file."""
    # from google.cloud import vision
    import io
    # client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    #print('Texts:')

    #text_bounding_box = {}

    ocr_text = []

    for text in texts:
        #print('\n"{}"'.format(text.description))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in text.bounding_poly.vertices])

        ocr_text.append(text.description)

        #print('bounds: {}'.format(','.join(vertices)))

#         print(text.description)
#         print(type(text.description))

#         x = []
#         y = []

#         for vertex in text.bounding_poly.vertices:
#             x.append(str(vertex.x))
#             y.append(str(vertex.y))

#         text_bounding_box[text.description]['x'] = x
#         text_bounding_box[text.description]['y'] = y


    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

        exception = str(response.error.message) + '\nFor more info on error messages, check: https://cloud.google.com/apis/design/errors'

        # client.transports.channel.close()
        # client._channel.close()

        return exception

    else:
        # client.transports.channel.close()
        # client._channel.close()
        return ocr_text#, text_bounding_box





# @application.route('/read_barcode', methods=['GET','POST'])
#
# def barcode_reader():
#
#     if request.method == 'POST':
#
#         if 'file' not in request.files:
#
#             return 'No file'
#
#         file = request.files['file']
#
#         if file.filename == '':
#
#             return 'No file'
#
#         if file and not allowed_file(file.filename):
#
#             return 'File is not of image format i.e jpg, jpeg, pdf...'
#
#         if file and allowed_file(file.filename):
#
#
#             #file.save(os.path.join('/home/ubuntu/Speech_Recognition/uploads', file.filename))
#             #file.save(os.path.join('/home/monu/speech_recognition/uploads', file.filename))
#             #file.save(os.path.join('/home/ubuntu/speech_recognition/speech_ai/uploads', file.filename))
#
#             #upload_folder = '/home/monu/speech_recognition/uploads/'
#             #upload_folder = '/home/monu/Mahindra_image_processing/uploads/'
#
#             # fn = file.filename
#             #
#             # file_name = upload_folder + fn
#             #
#             # file.save(os.path.join(upload_folder, fn))
#
#             time_stamp = '_' + str(time.time()).split('.')[0]
#
#             fn = file.filename
#
#             fn = fn.split('.')[0] + time_stamp + '.' + fn.split('.')[1]
#
#             file_name = upload_folder + fn
#
#             print('\n\nOriginal file name with time stamp: {}\n\n'.format(file_name) )
#
#             file.save(os.path.join(upload_folder, fn))
#
#
#             if file_name.split('.')[-1] == 'jpeg' or file_name.split('.')[-1] == 'jpg':# or name.split('.')[-1] == 'pdf':
#
#                 barcode = barcode_reader_class.barcode_reader(file_name)
#
#                 result = barcode
#
#
#             elif file_name.split('.')[-1] == 'pdf':# or name.split('.')[-1] == 'pdf':
#
#                 pages = convert_from_path(file_name, 500)
#
#                 page_no = 0
#
#                 for page in pages:
#
#                     page_no += 1
#                     new_file_name = file_name + 'out' + str(page_no)
#                     page.save(new_file_name, 'JPEG')
#
#                     barcode = barcode_reader_class.barcode_reader(new_file_name)
#
#                     result = barcode
#
#
#             return result
#
#
# @application.route('/pod_type', methods=['GET','POST'])
#
# def pod_classifier():
#
#     if request.method == 'POST':
#
#         if 'file' not in request.files:
#
#             return 'No file'
#
#         file = request.files['file']
#
#         if file.filename == '':
#
#             return 'No file'
#
#         if file and not allowed_file(file.filename):
#
#             return 'File is not of image format i.e jpg, jpeg, pdf...'
#
#         if file and allowed_file(file.filename):
#
#
#             #file.save(os.path.join('/home/ubuntu/Speech_Recognition/uploads', file.filename))
#             #file.save(os.path.join('/home/monu/speech_recognition/uploads', file.filename))
#             #file.save(os.path.join('/home/ubuntu/speech_recognition/speech_ai/uploads', file.filename))
#
#             #upload_folder = '/home/monu/speech_recognition/uploads/'
#             #upload_folder = '/home/monu/Mahindra_image_processing/uploads/'
#
#             # fn = file.filename
#             #
#             # file_name = upload_folder + fn
#             #
#             # file.save(os.path.join(upload_folder, fn))
#
#             time_stamp = '_' + str(time.time()).split('.')[0]
#
#             fn = file.filename
#
#             fn = fn.split('.')[0] + time_stamp + '.' + fn.split('.')[1]
#
#             file_name = upload_folder + fn
#
#             print('\n\nOriginal file name with time stamp: {}\n\n'.format(file_name) )
#
#             file.save(os.path.join(upload_folder, fn))
#
#
#
#             if file_name.split('.')[-1] == 'jpeg' or file_name.split('.')[-1] == 'jpg':# or name.split('.')[-1] == 'pdf':
#
#                 pod_type = pod_classifier_class.pod_classifier(file_name, invalid_words_unique, pod_words_pre_print_unique, printed_pod_extra_words_unique)
#
#                 result = pod_type
#
#             elif file_name.split('.')[-1] == 'pdf':# or name.split('.')[-1] == 'pdf':
#
#                 pages = convert_from_path(file_name, 500)
#
#                 page_no = 0
#
#                 for page in pages:
#
#                     page_no += 1
#                     new_file_name = file_name + 'out' + str(page_no)
#                     page.save(new_file_name, 'JPEG')
#
#                     pod_type = pod_classifier_class.pod_classifier(new_file_name, invalid_words_unique, pod_words_pre_print_unique, printed_pod_extra_words_unique)
#
#                     result = pod_type
#
#
#             return result
#
#
# @application.route('/complete_incomplete_pod', methods=['GET','POST'])
#
# def complete_incomplete_pod():
#
#     if request.method == 'POST':
#
#         if 'file' not in request.files:
#
#             return 'No file'
#
#         file = request.files['file']
#
#         if file.filename == '':
#
#             return 'No file'
#
#         if file and not allowed_file(file.filename):
#
#             return 'File is not of image format i.e jpg, jpeg, pdf...'
#
#         if file and allowed_file(file.filename):
#
#             #file.save(os.path.join('/home/ubuntu/Speech_Recognition/uploads', file.filename))
#             #file.save(os.path.join('/home/monu/speech_recognition/uploads', file.filename))
#             #file.save(os.path.join('/home/ubuntu/speech_recognition/speech_ai/uploads', file.filename))
#
#             #upload_folder = '/home/monu/speech_recognition/uploads/'
#             #upload_folder = '/home/monu/Mahindra_image_processing/uploads/'
#
#             # fn = file.filename
#             #
#             # file_name = upload_folder + fn
#             #
#             # file.save(os.path.join(upload_folder, fn))
#
#             time_stamp = '_' + str(time.time()).split('.')[0]
#
#             fn = file.filename
#
#             fn = fn.split('.')[0] + time_stamp + '.' + fn.split('.')[1]
#
#             file_name = upload_folder + fn
#
#             print('\n\nOriginal file name with time stamp: {}\n\n'.format(file_name) )
#
#             file.save(os.path.join(upload_folder, fn))
#
#
#
#
#             if file_name.split('.')[-1] == 'jpeg' or file_name.split('.')[-1] == 'jpg':# or name.split('.')[-1] == 'pdf':
#
#                 # pre_print_pod_complete_or_not, boundary_words_detected_pre_print = complete_incomplete_class.complete_pod_or_not_pre_print(file_name, boundary_words_pre_print, fifty_percent_of_total_pre_print_boundary_words)
#                 # result = pre_print_pod_complete_or_not
#
#                 printed_pod_complete_or_not, boundary_words_detected_printed = complete_incomplete_class.complete_pod_or_not_printed(file_name, boundary_words_printed, fifty_percent_of_total_printed_boundary_words)
#                 result = printed_pod_complete_or_not
#
#
#             elif file_name.split('.')[-1] == 'pdf':# or name.split('.')[-1] == 'pdf':
#
#                 pages = convert_from_path(file_name, 500)
#
#                 page_no = 0
#
#                 for page in pages:
#
#                     page_no += 1
#                     new_file_name = file_name + 'out' + str(page_no)
#                     page.save(new_file_name, 'JPEG')
#
#                     # pre_print_pod_complete_or_not, boundary_words_detected_pre_print = omplete_incomplete_class.complete_pod_or_not_pre_print(file_name, boundary_words_pre_print, fifty_percent_of_total_pre_print_boundary_words)
#                     # result = pre_print_pod_complete_or_not
#
#                     printed_pod_complete_or_not, boundary_words_detected_printed = complete_incomplete_class.complete_pod_or_not_printed(new_file_name, boundary_words_printed, fifty_percent_of_total_printed_boundary_words)
#                     result = printed_pod_complete_or_not
#
#
#
#             return result
#
#
# @application.route('/clear_or_unclear_pod', methods=['GET','POST'])
#
# def clear_unclear():
#
#     if request.method == 'POST':
#
#         if 'file' not in request.files:
#
#             return 'No file'
#
#         file = request.files['file']
#
#         if file.filename == '':
#
#             return 'No file'
#
#         if file and not allowed_file(file.filename):
#
#             return 'File is not of image format i.e jpg, jpeg, pdf...'
#
#         if file and allowed_file(file.filename):
#
#             #file.save(os.path.join('/home/ubuntu/Speech_Recognition/uploads', file.filename))
#             #file.save(os.path.join('/home/monu/speech_recognition/uploads', file.filename))
#             #file.save(os.path.join('/home/ubuntu/speech_recognition/speech_ai/uploads', file.filename))
#
#             #upload_folder = '/home/monu/speech_recognition/uploads/'
#             #upload_folder = '/home/monu/Mahindra_image_processing/uploads/'
#
#             # fn = file.filename
#             #
#             # file_name = upload_folder + fn
#             #
#             # file.save(os.path.join(upload_folder, fn))
#
#             time_stamp = '_' + str(time.time()).split('.')[0]
#
#             fn = file.filename
#
#             fn = fn.split('.')[0] + time_stamp + '.' + fn.split('.')[1]
#
#             file_name = upload_folder + fn
#
#             print('\n\nOriginal file name with time stamp: {}\n\n'.format(file_name) )
#
#             file.save(os.path.join(upload_folder, fn))
#
#
#
#             if file_name.split('.')[-1] == 'jpeg' or file_name.split('.')[-1] == 'jpg':# or name.split('.')[-1] == 'pdf':
#
#                 img_quality = clear_not_clear_class.detect_img_quality(file_name)
#
#                 brisque_img_quality = clear_not_clear_class.brisque_quality(file_name)
#
#                 printed_pod_clear_or_not, words_detected_printed = clear_not_clear_class.clear_pod_or_not_printed(file_name, pod_words_printed, fifty_percent_of_total_printed_words)
#
#                 result = {"img_quality": img_quality, "brisque_img_quality": brisque_img_quality, "printed_pod_clear_or_not": printed_pod_clear_or_not}
#
#
#
#             elif file_name.split('.')[-1] == 'pdf':# or name.split('.')[-1] == 'pdf':
#
#
#                 pages = convert_from_path(file_name, 500)
#
#                 page_no = 0
#
#                 for page in pages:
#
#                     page_no += 1
#                     new_file_name = file_name + 'out' + str(page_no)
#                     page.save(new_file_name, 'JPEG')
#
#                     img_quality = clear_not_clear_class.detect_img_quality(new_file_name)
#
#                     brisque_img_quality = clear_not_clear_class.brisque_quality(new_file_name)
#
#                     printed_pod_clear_or_not, words_detected_printed = clear_pod_or_not_printed(new_file_name, pod_words_printed, fifty_percent_of_total_printed_words)
#
#                     result = {"img_quality": img_quality, "brisque_img_quality": brisque_img_quality, "printed_pod_clear_or_not": printed_pod_clear_or_not}
#
#
#             return result


@application.route('/hello_world', methods=['GET','POST'])

def hello_world():

    return 'Hello World'


@application.route('/clear_unclear_decision', methods=['GET','POST'])

def decision_clear_or_not():

    overall_start = time.time()
    if request.method == 'POST':
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
        #
        #     #file.save(os.path.join('/home/ubuntu/Speech_Recognition/uploads', file.filename))
        #     #file.save(os.path.join('/home/monu/speech_recognition/uploads', file.filename))
        #     #file.save(os.path.join('/home/ubuntu/speech_recognition/speech_ai/uploads', file.filename))
        #
        #     #upload_folder = '/home/monu/speech_recognition/uploads/'
        #     #upload_folder = '/home/monu/Mahindra_image_processing/uploads/'
        #
        #     # fn = file.filename
        #     #
        #     # file_name = upload_folder + fn
        #     #
        #     # file.save(os.path.join(upload_folder, fn))
        #
        #     time_stamp = '_' + str(time.time()).split('.')[0]
        #
        #     fn = file.filename
        #
        #     fn = fn.split('.')[0] + time_stamp + '.' + fn.split('.')[1]
        #
        #     file_name = upload_folder + fn
        #
        #     print('\n\nOriginal file name with time stamp: {}\n\n'.format(file_name) )
        #
        #     file.save(os.path.join(upload_folder, fn))
#-------------------------------------------------------------------------------------------------------

#-------------------------------------------

        try:

            json_file = request.json

            print('json file: {}'.format(json_file))
            logger.info('json file: {}'.format(json_file))


            filename = json_file['image_url']

            print('Original filename: {}'.format(filename))
            logger.info('Original filename: {}'.format(filename))
    #-------------------------------------------

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
    #--------------------------------
            if filename == '':

                print('No File')
                logger.info('No File')

                return 'No file'
    #--------------------------------
            # if allowed_file(filename):
            #
            #     return 'File is not of image format i.e jpg, jpeg, pdf...'
    #------------------------------------------------------------------------------------------------------------
            elif allowed_file(filename):



                #file.save(os.path.join('/home/ubuntu/Speech_Recognition/uploads', file.filename))
                #file.save(os.path.join('/home/monu/speech_recognition/uploads', file.filename))
                #file.save(os.path.join('/home/ubuntu/speech_recognition/speech_ai/uploads', file.filename))

                #upload_folder = '/home/monu/speech_recognition/uploads/'
                #upload_folder = '/home/monu/Mahindra_image_processing/uploads/'

                # fn = file.filename
                #
                # file_name = upload_folder + fn
                #
                # file.save(os.path.join(upload_folder, fn))

                #time_stamp = '_' + str(time.time()).split('.')[0]

                # fn = file.filename
                fn = filename.split('/')[-1]

                #fn = fn.split('.')[0] + time_stamp + '.' + fn.split('.')[1]


                file_name = upload_folder + fn

                print('\n\nOriginal file name with time stamp: {}\n\n'.format(file_name))

                #file.save(os.path.join(upload_folder, fn))
                urllib.request.urlretrieve(filename, file_name)

                logger.info("\n\nOriginal file name with time stamp: {}\n\n".format(file_name.split('/')[-1]))

    #-------------------------------------------------------------------------------------------------------------


                if file_name.split('.')[-1] == 'jpeg' or file_name.split('.')[-1] == 'jpg':# or name.split('.')[-1] == 'pdf':

                    ocr_start = time.time()

                    try:
                        ocr_text = detect_text(file_name)

                    except Exception as e:

                        print('Exception: {}'.format(e))
                        logger.info('Exception: {}'.format(e))

                        logging.exception("Exception occurred", exc_info=True)

                        ocr_text = detect_text(file_name)



                    ocr_end = time.time()
                    print('Time taken for OCR response: {}'.format(ocr_end - ocr_start))

                    logger.info('Time taken for OCR response: {}'.format(ocr_end - ocr_start))


                    barcode_start = time.time()
                    barcode = barcode_reader_class.barcode_reader(file_name, ocr_text)
                    barcode_end = time.time()
                    print(barcode)

                    logger.info(barcode)


                    print('Time taken for Barcode reader: {}'.format(barcode_end - barcode_start))

                    logger.info('Time taken for Barcode reader: {}'.format(barcode_end - barcode_start))

                    #result = barcode
                    pod_start = time.time()
                    pod_type = pod_classifier_class.pod_classifier(file_name, ocr_text, invalid_words_unique, pod_words_pre_print_unique, printed_pod_extra_words_unique)
                    pod_end = time.time()

                    print('pod type: {}'.format(pod_type))

                    logger.info('pod type: {}'.format(pod_type))

                    print('Time taken for detecting POD type: {}'.format(pod_end - pod_start))

                    logger.info('Time taken for detecting POD type: {}'.format(pod_end - pod_start))

                    invoice_start = time.time()
                    invoice_no = get_invoice_no_class.invoice_no_detector(file_name, ocr_text)
                    invoice_end = time.time()

                    print('invoice type: ', type(invoice_no))
                    print('invoice length: ', len(invoice_no))
                    print('Invoice no: ', invoice_no)
                    logger.info('Invoice no: '.format(invoice_no))

                    print('Time taken for Invoice number detection: {}'.format(invoice_end - invoice_start))

                    logger.info('Time taken for Invoice number detection: {}'.format(invoice_end - invoice_start))

                    stamp_start = time.time()
                    stamp_detection = stamp_detection_class.stamp_clear_or_not(file_name, ocr_text, pod_words_printed_consignee)
                    stamp_end = time.time()
                    print('Time taken for detecting the stamp: {}'.format(stamp_end - stamp_start))

                    logger.info('Time taken for detecting the stamp: {}'.format(stamp_end - stamp_start))

                    print('stamp detection: {}'.format(stamp_detection))
                    logger.info('stamp detection: {}'.format(stamp_detection))

                    logo_start = time.time()

                    logo = ''

                    for i, wrd in enumerate(ocr_text):


                        no_of_mistakes = nltk.edit_distance('LOGI', wrd[0:3])#'LOGISTICS', wrd

                        if no_of_mistakes <= 2:

                            logo = 'present'

                            break

                        else:

                            no_of_mistakes = nltk.edit_distance('TICS', wrd[-4:])#'LOGISTICS', wrd

                            if no_of_mistakes <= 2:

                                logo = 'present'

                                break

                            else:

                                logo = 'absent'


                    if logo == '':
                        logo = 'absent'

                    print('Logo: {}'.format(logo))
                    logger.info('Logo: {}'.format(logo))

                    logo_end = time.time()
                    print('Time taken for Logo detection: {}'.format(logo_end - logo_start))

                    logger.info('Time taken for Logo detection: {}'.format(logo_end - logo_start))

                    if pod_type != 'Invalid POD' and pod_type != 'Invalid POD: 2 copies found' and pod_type != 'Invalid/Unclear POD' and barcode != 'Barcode not found' and barcode != 'Barcode not extracted' and barcode is not None and logo == 'present' and len(invoice_no) >= 0 and stamp_detection == 'stamp detected':

                        print('I am in if block')
                        if pod_type == 'Pre-print POD':

                            complete_start = time.time()
                            pre_print_pod_complete_or_not, boundary_words_detected_pre_print = complete_incomplete_class.complete_pod_or_not_pre_print(file_name, ocr_text, boundary_words_pre_print, fifty_percent_of_total_pre_print_boundary_words)
                            complete_end = time.time()
                            print('Time taken for detecting complete or incomplete POD: {}'.format(complete_end - complete_start))
                            logger.info('Time taken for detecting complete or incomplete POD: {}'.format(complete_end - complete_start))

                            img_quality_start = time.time()
                            img_quality = clear_not_clear_class.detect_img_quality(file_name)
                            img_quality_end = time.time()
                            print('Time taken for detecting image quality i.e contrast, blurriness and brightness: {}'.format(img_quality_end - img_quality_start))
                            logger.info('Time taken for detecting image quality i.e contrast, blurriness and brightness: {}'.format(img_quality_end - img_quality_start))

                            brisque_img_quality_start = time.time()
                            brisque_img_quality = clear_not_clear_class.brisque_quality(file_name)
                            brisque_img_quality_end = time.time()
                            print('Time taken for detcting BRISQUE image quality: {}'.format(brisque_img_quality_end - brisque_img_quality_start))
                            logger.info('Time taken for detcting BRISQUE image quality: {}'.format(brisque_img_quality_end - brisque_img_quality_start))

                            #pre_print_pod_clear_or_not, words_detected_pre_print = clear_not_clear_class.clear_pod_or_not_pre_print(file_name, pod_words_pre_print, fifty_percent_of_total_pre_print_words)

                            reasons = []
#######################################################################################################################
                            # if img_quality == 'Low contrast image':
                            #
                            #     reasons.append(img_quality)
                            #
                            # if img_quality == 'Blurry image':
                            #
                            #     reasons.append(img_quality)
                            #
                            # if img_quality == 'Dark image':
                            #
                            #     reasons.append(img_quality)
                            #
                            # if brisque_img_quality < 10:
                            #
                            #     reasons.append("Overall image quality(BRISQUE) is too low")
############################################################################################################################

                            if pre_print_pod_complete_or_not == 'Incomplete POD':

                                reasons.append(pre_print_pod_complete_or_not)

                            if len(reasons) == 0:

                                reasons.append("No error found")

                            #reasons = creating_reasons(reasons)
#####################################################################################################
                            # if img_quality != 'Low contrast image' and img_quality != 'Blurry image' and img_quality != 'Dark image' and brisque_img_quality > 10 and pre_print_pod_complete_or_not != 'Incomplete POD':#and pre_print_pod_clear_or_not != 'Unclear POD'
                            #
                            #     result = {"clear_unclear_prediction": "Clear","pod_type": pod_type, "barcode": barcode, "img_quality": img_quality, "brisque_img_quality": brisque_img_quality, "pre_print_pod_complete_or_not": pre_print_pod_complete_or_not, "invoice_no": invoice_no, 'stamp detected': stamp_detection, 'logo': logo, "reasons": reasons}#, "pre_print_pod_clear_or_not": pre_print_pod_clear_or_not
                            #
                            # else:
                            #
                            #     result = {"clear_unclear_prediction": "Unclear","pod_type": pod_type, "barcode": barcode, "img_quality": img_quality, "brisque_img_quality": brisque_img_quality, "pre_print_pod_complete_or_not": pre_print_pod_complete_or_not, "invoice_no": invoice_no, 'stamp detected': stamp_detection, 'logo': logo, "reasons": reasons}#, "pre_print_pod_clear_or_not": pre_print_pod_clear_or_not
#####################################################################################################

                            if pre_print_pod_complete_or_not != 'Incomplete POD':#and pre_print_pod_clear_or_not != 'Unclear POD'

                                result = {"clear_unclear_prediction": "Clear","pod_type": pod_type, "barcode": barcode, "pre_print_pod_complete_or_not": pre_print_pod_complete_or_not, 'stamp detected': stamp_detection, 'logo': logo, "reasons": reasons}#, "pre_print_pod_clear_or_not": pre_print_pod_clear_or_not

                            else:

                                result = {"clear_unclear_prediction": "Unclear","pod_type": pod_type, "barcode": barcode, "pre_print_pod_complete_or_not": pre_print_pod_complete_or_not, 'stamp detected': stamp_detection, 'logo': logo, "reasons": reasons}#, "pre_print_pod_clear_or_not": pre_print_pod_clear_or_not



                        elif pod_type == 'Printed POD':

                            #result = pod_type
                            complete_start = time.time()
                            printed_pod_complete_or_not, boundary_words_detected_printed = complete_incomplete_class.complete_pod_or_not_printed(file_name, ocr_text, boundary_words_printed, fifty_percent_of_total_printed_boundary_words)
                            complete_end = time.time()
                            print('Time taken for detecting complete or incomplete POD: {}'.format(complete_end - complete_start))

                            logger.info('Time taken for detecting complete or incomplete POD: {}'.format(complete_end - complete_start))

                            img_quality_start = time.time()
                            img_quality = clear_not_clear_class.detect_img_quality(file_name)
                            img_quality_end = time.time()
                            print('Time taken for detecting image quality i.e contrast, blurriness and brightness: {}'.format(img_quality_end - img_quality_start))

                            logger.info('Time taken for detecting image quality i.e contrast, blurriness and brightness: {}'.format(img_quality_end - img_quality_start))

                            brisque_img_quality_start = time.time()
                            brisque_img_quality = clear_not_clear_class.brisque_quality(file_name)
                            brisque_img_quality_end = time.time()
                            print('Time taken for detcting BRISQUE image quality: {}'.format(brisque_img_quality_end - brisque_img_quality_start))

                            logger.info('Time taken for detcting BRISQUE image quality: {}'.format(brisque_img_quality_end - brisque_img_quality_start))

                            #printed_pod_clear_or_not, words_detected_printed = clear_not_clear_class.clear_pod_or_not_printed(file_name, pod_words_printed, fifty_percent_of_total_printed_words)

                            reasons = []
##############################################################################################
                            # if img_quality == 'Low contrast image':
                            #
                            #     reasons.append(img_quality)
                            #
                            # if img_quality == 'Blurry image':
                            #
                            #     reasons.append(img_quality)
                            #
                            # if img_quality == 'Dark image':
                            #
                            #     reasons.append(img_quality)
                            #
                            # if brisque_img_quality < 10:
                            #
                            #     reasons.append("Overall image quality(BRISQUE) is too low")
#############################################################################################
                            if printed_pod_complete_or_not == 'Incomplete POD':

                                reasons.append(printed_pod_complete_or_not)

                            if len(reasons) == 0:

                                reasons.append("No error found")

                            #reasons = creating_reasons(reasons)
##############################################################################################
                            # if img_quality != 'Low contrast image' and img_quality != 'Blurry image' and img_quality != 'Dark image' and brisque_img_quality > 10 and printed_pod_complete_or_not != 'Incomplete POD':#and printed_pod_clear_or_not != 'Unclear POD'
                            #
                            #     result = {"clear_unclear_prediction": "Clear","pod_type": pod_type, "barcode": barcode, "img_quality": img_quality, "brisque_img_quality": brisque_img_quality, "printed_pod_complete_or_not": printed_pod_complete_or_not, "invoice_no": invoice_no, 'stamp detected': stamp_detection, 'logo': logo, "reasons": reasons}#, "printed_pod_clear_or_not": printed_pod_clear_or_not
                            #
                            # else:
                            #
                            #     result = {"clear_unclear_prediction": "Unclear","pod_type": pod_type, "barcode": barcode, "img_quality": img_quality, "brisque_img_quality": brisque_img_quality, "printed_pod_complete_or_not": printed_pod_complete_or_not, "invoice_no": invoice_no, 'stamp detected': stamp_detection, 'logo': logo, "reasons": reasons}#, "printed_pod_clear_or_not": printed_pod_clear_or_not
##################################################################################################

                            if printed_pod_complete_or_not != 'Incomplete POD':#and printed_pod_clear_or_not != 'Unclear POD'

                                result = {"clear_unclear_prediction": "Clear","pod_type": pod_type, "barcode": barcode, "printed_pod_complete_or_not": printed_pod_complete_or_not, 'stamp detected': stamp_detection, 'logo': logo, "reasons": reasons}#, "printed_pod_clear_or_not": printed_pod_clear_or_not

                            else:

                                result = {"clear_unclear_prediction": "Unclear","pod_type": pod_type, "barcode": barcode, "printed_pod_complete_or_not": printed_pod_complete_or_not, 'stamp detected': stamp_detection, 'logo': logo, "reasons": reasons}#, "printed_pod_clear_or_not": printed_pod_clear_or_not

                    else:

                        print('I am in else block')

                        reasons = []


                        if pod_type == 'Invalid POD':

                            reasons.append(pod_type)

                        if pod_type == 'Invalid POD: 2 copies found':

                            reasons.append('Invalid POD: 2 PODs found in the image')

                        if pod_type == 'Invalid/Unclear POD':

                            reasons.append('Invalid/Unclear')

                        if barcode == 'Barcode not found':

                            reasons.append("LR number not found in image")

                        if barcode == 'Barcode not extracted':

                            reasons.append('Barcode is not readable')

                        if logo == 'absent':

                            reasons.append('Logo is Missing')

                        # if len(invoice_no) == 0:
                        #
                        #     reasons.append('Invoice number not found')

                        if stamp_detection == 'stamp not detected':

                            reasons.append('Stamp either missing or unclear')

                        if len(reasons) == 0:

                            reasons.append("No error found")

                        #reasons = creating_reasons(reasons)


                        print('invoice type: '.format(type(invoice_no)))
                        print('invoice length: '.format(len(invoice_no)))
                        print('Invoice no: '.format(invoice_no))



                        result = {"clear_unclear_prediction": "Invalid/Unclear","pod_type": pod_type, "barcode": barcode, 'stamp detected': stamp_detection, 'logo': logo, "reasons": reasons}#"invoice_no": invoice_no,

                        print('inner result: {}'.format(result))
                    logger.info("result: {}".format(result))
                    logger.info('\n--------------------------------------------------------------------------------------------------------------------------\n')

                    result = json.dumps(result)

                    print('\n')
                    print(result)
                    print('\n')

                    overall_end = time.time()
                    print('Overall time taken for deciding clear or not clear POD: {}'.format(overall_end - overall_start))

                    logger.info('Overall time taken for deciding clear or not clear POD: {}'.format(overall_end - overall_start))

                    if os.path.exists(file_name):

                      os.remove(file_name)

                      logger.info('Deleting the file: {}'.format(file_name))
                      print('Deleting the file: {}'.format(file_name))

                    else:

                      print("The file does not exist: {}".format(file_name))

                    return result

                elif file_name.split('.')[-1] == 'pdf':# or name.split('.')[-1] == 'pdf':

                    pages = convert_from_path(file_name, 500)

                    page_no = 0

                    for page in pages:

                        pdf_start = time.time()

                        page_no += 1
                        new_file_name = file_name.split('.')[0] + '_out_' + str(page_no) + '.jpg'
                        page.save(new_file_name, 'JPEG')

                        pdf_end = time.time()
                        print('Time taken for converting from pdf to JPEG for page number {}: {}'.format(page_no, (pdf_end - pdf_start)))

                        logger.info('Time taken for converting from pdf to JPEG for page number {}: {}'.format(page_no, (pdf_end - pdf_start)))


##################################################################################################################################################
                        ocr_start = time.time()

                        try:
                            ocr_text = detect_text(new_file_name)

                        except Exception as e:
                            print('Exception: '.format(e))
                            logger.info('Exception: '.format(e))

                            ocr_text = detect_text(new_file_name)

                            logging.exception("Exception occurred", exc_info=True)


                        ocr_end = time.time()
                        print('Time taken for OCR response: {}'.format(ocr_end - ocr_start))

                        logger.info('Time taken for OCR response: {}'.format(ocr_end - ocr_start))

                        barcode_start = time.time()
                        barcode = barcode_reader_class.barcode_reader(new_file_name, ocr_text)
                        barcode_end = time.time()

                        print(barcode)

                        logger.info(barcode)

                        print('Time taken for Barcode reader: {}'.format(barcode_end - barcode_start))

                        logger.info('Time taken for Barcode reader: {}'.format(barcode_end - barcode_start))

                        #result = barcode
                        pod_start = time.time()
                        pod_type = pod_classifier_class.pod_classifier(new_file_name, ocr_text, invalid_words_unique, pod_words_pre_print_unique, printed_pod_extra_words_unique)
                        pod_end = time.time()
                        print('Time taken for detecting POD type: {}'.format(pod_end - pod_start))

                        logger.info('Time taken for detecting POD type: {}'.format(pod_end - pod_start))

                        print('Pod type: {}'.format(pod_type))

                        logger.info('Pod type: {}'.format(pod_type))

                        invoice_start = time.time()
                        invoice_no = get_invoice_no_class.invoice_no_detector(new_file_name, ocr_text)
                        invoice_end = time.time()
                        print('Time taken for Invoice number detection: {}'.format(invoice_end - invoice_start))

                        logger.info('Time taken for Invoice number detection: {}'.format(invoice_end - invoice_start))

                        print('invoice_no: {}'.format(invoice_no))

                        logger.info('invoice_no: {}'.format(invoice_no))

                        stamp_start = time.time()
                        stamp_detection = stamp_detection_class.stamp_clear_or_not(new_file_name, ocr_text, pod_words_printed_consignee)
                        stamp_end = time.time()
                        print('Time taken for detecting the stamp: {}'.format(stamp_end - stamp_start))

                        logger.info('Time taken for detecting the stamp: {}'.format(stamp_end - stamp_start))


                        print('stamp detection: {}'.format(stamp_detection))

                        logger.info('stamp detection: {}'.format(stamp_detection))

                        logo_start = time.time()

                        logo = ''

                        for i, wrd in enumerate(ocr_text):


                            no_of_mistakes = nltk.edit_distance('LOGI', wrd[0:3])#'LOGISTICS', wrd

                            if no_of_mistakes <= 2:

                                logo = 'present'

                                break

                            else:

                                logo = 'absent'

                        if logo == '':
                            logo = 'absent'


                        logo_end = time.time()

                        print('Logo: {}'.format(logo))

                        logger.info('Logo: {}'.format(logo))

                        print('Time taken for Logo detection: {}'.format(logo_end - logo_start))

                        logger.info('Time taken for Logo detection: {}'.format(logo_end - logo_start))


                        if pod_type != 'Invalid POD' and pod_type != 'Invalid POD: 2 copies found' and pod_type != 'Invalid/Unclear POD' and barcode != 'Barcode not found' and barcode != 'Barcode not extracted' and barcode is not None and logo == 'present' and len(invoice_no) >= 0 and stamp_detection == 'stamp detected':#

                            if pod_type == 'Pre-print POD':

                                complete_start = time.time()
                                pre_print_pod_complete_or_not, boundary_words_detected_pre_print = complete_incomplete_class.complete_pod_or_not_pre_print(new_file_name, ocr_text, boundary_words_pre_print, fifty_percent_of_total_pre_print_boundary_words)
                                complete_end = time.time()
                                print('Time taken for detecting complete or incomplete POD: {}'.format(complete_end - complete_start))

                                logger.info('Time taken for detecting complete or incomplete POD: {}'.format(complete_end - complete_start))

                                img_quality = clear_not_clear_class.detect_img_quality(new_file_name)

                                #brisque_img_quality = clear_not_clear_class.brisque_quality(new_file_name)

                                #pre_print_pod_clear_or_not, words_detected_pre_print = clear_not_clear_class.clear_pod_or_not_pre_print(new_file_name, pod_words_pre_print, fifty_percent_of_total_pre_print_words)

                                reasons = []
###################################################################################################
                                # if img_quality == 'Low contrast image':
                                #
                                #     reasons.append(img_quality)
                                #
                                # if img_quality == 'Blurry image':
                                #
                                #     reasons.append(img_quality)
                                #
                                # if img_quality == 'Dark image':
                                #
                                #     reasons.append(img_quality)
########################################################################################################
                                # if brisque_img_quality < 10:
                                #
                                #     reasons.append("Overall image quality(BRISQUE) is too low")

                                if pre_print_pod_complete_or_not == 'Incomplete POD':

                                    reasons.append(pre_print_pod_complete_or_not)

                                if len(reasons) == 0:

                                    reasons.append("No error found")

                                # if pre_print_pod_complete_or_not == 'Incomplete POD':
                                #
                                #     reasons.append(pre_print_pod_complete_or_not)
                                #
                                # else:
                                #
                                #     reasons.append("No error found")


##################################################################################################################
                                # if img_quality != 'Low contrast image' and img_quality != 'Blurry image' and img_quality != 'Dark image' and pre_print_pod_complete_or_not != 'Incomplete POD':#and brisque_img_quality > 10 #and pre_print_pod_clear_or_not != 'Unclear POD'
                                #
                                #     result = {"clear_unclear_prediction": "Clear","pod_type": pod_type, "barcode": barcode, "img_quality": img_quality, "brisque_img_quality": 'Not detecting', "pre_print_pod_complete_or_not": pre_print_pod_complete_or_not, "invoice_no": invoice_no, 'stamp detected': stamp_detection, 'logo': logo, "reasons": reasons}#, "pre_print_pod_clear_or_not": pre_print_pod_clear_or_not
                                #
                                # else:
                                #
                                #     result = {"clear_unclear_prediction": "Unclear","pod_type": pod_type, "barcode": barcode, "img_quality": img_quality, "brisque_img_quality": 'Not detecting', "pre_print_pod_complete_or_not": pre_print_pod_complete_or_not, "invoice_no": invoice_no, 'stamp detected': stamp_detection, 'logo': logo, "reasons": reasons}#, "pre_print_pod_clear_or_not": pre_print_pod_clear_or_not
###################################################################################################################

                                if pre_print_pod_complete_or_not != 'Incomplete POD':#and brisque_img_quality > 10 #and pre_print_pod_clear_or_not != 'Unclear POD'

                                    result = {"clear_unclear_prediction": "Clear","pod_type": pod_type, "barcode": barcode, "pre_print_pod_complete_or_not": pre_print_pod_complete_or_not, 'stamp detected': stamp_detection, 'logo': logo, "reasons": reasons}#, "pre_print_pod_clear_or_not": pre_print_pod_clear_or_not

                                else:

                                    result = {"clear_unclear_prediction": "Unclear","pod_type": pod_type, "barcode": barcode, "pre_print_pod_complete_or_not": pre_print_pod_complete_or_not, 'stamp detected': stamp_detection, 'logo': logo, "reasons": reasons}#, "pre_print_pod_clear_or_not": pre_print_pod_clear_or_not




                                # if img_quality != 'Overall quality: Bad' and brisque_img_quality > 10 and pre_print_pod_clear_or_not != 'Unclear POD' and pre_print_pod_complete_or_not != 'Incomplete POD':#
                                #
                                #     result = {"clear_unclear_prediction": "Clear","pod_type": pod_type, "barcode": barcode, "pre_print_pod_complete_or_not": pre_print_pod_complete_or_not, "invoice_no": invoice_no, 'stamp detected': stamp_detection, 'logo': logo, "reasons": reasons}#, "pre_print_pod_clear_or_not": pre_print_pod_clear_or_not
                                #     #"brisque_img_quality": brisque_img_quality,  "img_quality": img_quality,
                                # else:
                                #
                                #     result = {"clear_unclear_prediction": "Unclear","pod_type": pod_type, "barcode": barcode, "pre_print_pod_complete_or_not": pre_print_pod_complete_or_not, "invoice_no": invoice_no, 'stamp detected': stamp_detection, 'logo': logo, "reasons": reasons}#, "pre_print_pod_clear_or_not": pre_print_pod_clear_or_not
                                #     #"brisque_img_quality": brisque_img_quality,  "img_quality": img_quality,


                            elif pod_type == 'Printed POD':

                                #result = pod_type
                                complete_start = time.time()
                                printed_pod_complete_or_not, boundary_words_detected_printed = complete_incomplete_class.complete_pod_or_not_printed(new_file_name, ocr_text, boundary_words_printed, fifty_percent_of_total_printed_boundary_words)
                                complete_end = time.time()
                                print('Time taken for detecting complete or incomplete POD: {}'.format(complete_end - complete_start))

                                logger.info('Time taken for detecting complete or incomplete POD: {}'.format(complete_end - complete_start))

                                img_quality = clear_not_clear_class.detect_img_quality(new_file_name)

                                #brisque_img_quality = clear_not_clear_class.brisque_quality(new_file_name)

                                #printed_pod_clear_or_not, words_detected_printed = clear_not_clear_class.clear_pod_or_not_printed(new_file_name, pod_words_printed, fifty_percent_of_total_printed_words)

                                reasons = []
####################################################################################################
                                # if img_quality == 'Low contrast image':
                                #
                                #     reasons.append(img_quality)
                                #
                                # if img_quality == 'Blurry image':
                                #
                                #     reasons.append(img_quality)
                                #
                                # if img_quality == 'Dark image':
                                #
                                #     reasons.append(img_quality)
######################################################################################################

                                # if brisque_img_quality < 10:
                                #
                                #     reasons.append("Overall image quality(BRISQUE) is too low")

                                if printed_pod_complete_or_not == 'Incomplete POD':

                                    reasons.append(pre_print_pod_complete_or_not)

                                if len(reasons) == 0:

                                    reasons.append("No error found")


                                # if img_quality != 'Overall quality: Bad':
                                #
                                #     reasons.append(img_quality)
                                #
                                # elif brisque_img_quality > 10:
                                #
                                #     reasons.append(brisque_img_quality)

                                # if printed_pod_complete_or_not == 'Incomplete POD':
                                #
                                #     reasons.append(printed_pod_complete_or_not)
                                #
                                # else:
                                #
                                #     reasons.append("No error found")

##################################################################################################################
                                # if img_quality != 'Low contrast image' and img_quality != 'Blurry image' and img_quality != 'Dark image' and printed_pod_complete_or_not != 'Incomplete POD':#and brisque_img_quality > 10 #and printed_pod_clear_or_not != 'Unclear POD'
                                #
                                #     result = {"clear_unclear_prediction": "Clear","pod_type": pod_type, "barcode": barcode, "img_quality": img_quality, "brisque_img_quality": 'Not detecting', "printed_pod_complete_or_not": printed_pod_complete_or_not, "invoice_no": invoice_no, 'stamp detected': stamp_detection, 'logo': logo, "reasons": reasons}#, "printed_pod_clear_or_not": printed_pod_clear_or_not
                                #
                                # else:
                                #
                                #     result = {"clear_unclear_prediction": "Unclear","pod_type": pod_type, "barcode": barcode, "img_quality": img_quality, "brisque_img_quality": 'Not detecting', "printed_pod_complete_or_not": printed_pod_complete_or_not, "invoice_no": invoice_no, 'stamp detected': stamp_detection, 'logo': logo, "reasons": reasons}#, "printed_pod_clear_or_not": printed_pod_clear_or_not
####################################################################################################################


                                if printed_pod_complete_or_not != 'Incomplete POD':#and brisque_img_quality > 10 #and printed_pod_clear_or_not != 'Unclear POD'

                                    result = {"clear_unclear_prediction": "Clear","pod_type": pod_type, "barcode": barcode, "printed_pod_complete_or_not": printed_pod_complete_or_not, 'stamp detected': stamp_detection, 'logo': logo, "reasons": reasons}#, "printed_pod_clear_or_not": printed_pod_clear_or_not

                                else:

                                    result = {"clear_unclear_prediction": "Unclear","pod_type": pod_type, "barcode": barcode, "printed_pod_complete_or_not": printed_pod_complete_or_not, 'stamp detected': stamp_detection, 'logo': logo, "reasons": reasons}#, "printed_pod_clear_or_not": printed_pod_clear_or_not




                                # if printed_pod_complete_or_not != 'Incomplete POD':#img_quality != 'Overall quality: Bad' and brisque_img_quality > 10 and printed_pod_clear_or_not != 'Unclear POD' and
                                #
                                #     result = {"clear_unclear_prediction": "Clear","pod_type": pod_type, "barcode": barcode, "printed_pod_complete_or_not": printed_pod_complete_or_not, "invoice_no": invoice_no, 'stamp detected': stamp_detection, 'logo': logo, "reasons": reasons}#, "printed_pod_clear_or_not": printed_pod_clear_or_not
                                #     #"brisque_img_quality": brisque_img_quality, "img_quality": img_quality,
                                # else:
                                #
                                #
                                #
                                #     result = {"clear_unclear_prediction": "Unclear","pod_type": pod_type, "barcode": barcode, "printed_pod_complete_or_not": printed_pod_complete_or_not, "invoice_no": invoice_no, 'stamp detected': stamp_detection, 'logo': logo, "reasons": reasons}#, "printed_pod_clear_or_not": printed_pod_clear_or_not
                                #     #"brisque_img_quality": brisque_img_quality,  "img_quality": img_quality,
                        else:

                            reasons = []


                            if pod_type == 'Invalid POD':

                                reasons.append(pod_type)

                            if pod_type == 'Invalid POD: 2 copies found':

                                reasons.append('Invalid POD: 2 PODs found in the image')

                            if pod_type == 'Invalid/Unclear POD':

                                reasons.append(pod_type)

                            if barcode == 'Barcode not found':

                                reasons.append("LR number not found in image")

                            if barcode == 'Barcode not extracted':

                                reasons.append('Barcode is not readable')

                            # elif barcode == 'Barcode not found in image':
                            #
                            #     reasons.append(barcode)
                            #
                            # elif barcode is None:
                            #
                            #     reasons.append('Barcode reader not able to extract barcode')

                            if logo == 'absent':

                                reasons.append('Logo is Missing')

                            # if len(invoice_no) == 0:
                            #
                            #     reasons.append('Invoice number not found')

                            if stamp_detection == 'stamp not detected':

                                reasons.append('Stamp either missing or unclear')

                            if len(reasons) == 0:

                                reasons.append("No error found")

                            #reasons = creating_reasons(reasons)


                            result = {"clear_unclear_prediction": "Invalid/Unclear","pod_type": pod_type, "barcode": barcode, 'stamp detected': stamp_detection, 'logo': logo, "reasons": reasons}#"invoice_no": invoice_no,

                        logger.info("result: {}".format(result))
                        logger.info('\n--------------------------------------------------------------------------------------------------------------------------\n')

                        #result = json.dumps(result)
                        print('\n')
                        print(result)
                        print('\n')

                        overall_end = time.time()
                        print('Overall time taken for deciding clear or not clear POD: {}'.format(overall_end - overall_start))

                        logger.info('Overall time taken for deciding clear or not clear POD: {}'.format(overall_end - overall_start))

                        if os.path.exists(new_file_name):

                          os.remove(new_file_name)

                          logger.info('Deleting the file: {}'.format(new_file_name))
                          print('Deleting the file: {}'.format(new_file_name))

                        else:

                          print("The file does not exist: {}".format(new_file_name))

                        return result

            else:

                print('Error in image: Plz check')

                logger.info('Error in image: Plz check')

                return 'Error in image: Plz check'

        except Exception as e:

            print('Exception occured: {}'.format(e))

            exception = 'Exception occured: \n\n' + str(e)

            logger.info(exception)

            logging.exception("Exception occurred", exc_info=True)

            return exception


if __name__ == "__main__":

    application.run(host = '0.0.0.0', port='1234')
