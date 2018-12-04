#from inference_test import inference
#from batch_inference_test import batch_inference
from inference import inference
#from wer_metric import wer
import base64,io,tempfile
import logging
from falcon_cors import CORS
import falcon
import json
import traceback

logger=logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
#creating a file handler
handler=logging.FileHandler('tox21.log')
handler.setLevel(logging.DEBUG)
#Creating logging format
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)

class Icml(object):
   """Class serves the calls for transcription of the recordings"""
   def __init__(self):
       pass
   def on_get(self,req,r2esp):
       resp.body=json.dumps({'error':'Unsupported Request method'})
       resp.state=falcon.HTTP_501
   def on_post(self,req,resp):
       flag=True
       #logger.info('received request')
       try:
           name  = request.POST['File']
           result=inference(name)
           print(result)
           resp.body = json.dumps({'output':result})
           resp.state = falcon.HTTP_200
           #logger.info('received processed Sucessfully')
           #return JsonResponse()
       except Exception as e:
           print(e)
           traceback.print_exc()
           # logging.info(e)
           resp.body=json.dumps({'error':'unable to decode request; Request Not Acceptable!'})
           resp.state=falcon.HTTP_406
       
#APP definations
cors=CORS(allow_all_origins=True)
#cors=CORS( allow_all_origins=True,allow_all_headers=True,allow_all_methods=True)
APP = falcon.API(middleware=[cors.middleware])

#APP url definations

APP.add_route('/Icml',Icml())
#APP.add_route('/asr/batch_transcribe',BatchTrascriptionResource())
