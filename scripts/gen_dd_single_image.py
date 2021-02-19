import cv2
from dd_client import DD
import numpy as np
import argparse

host = 'localhost'
port = 8080
dd = DD(host,port)
dd.set_return_format(dd.RETURN_PYTHON)

parser = argparse.ArgumentParser()
parser.add_argument('--model-in-path',help='directory path that contains model to export (i.e. the .pt file)',required=True)
parser.add_argument('--img-size',default=256,type=int,help='square image size')
parser.add_argument('--img-in',help='image to transform',required=True)
parser.add_argument('--img-out',help='transformed image',required=True)
parser.add_argument('--gpu',help='whether to run on GPU',action='store_true')
args = parser.parse_args()

# service creation call
model = {
    'repository':args.model_in_path
    }
parameters_input = {
    'connector': 'image',
    'width': args.img_size,
    'height': args.img_size
}
parameters_mllib = {'gpu': args.gpu}
parameters_output = {}
try:    
    jout = dd.put_service('testggan',model,'gan generator inference test',
                          'torch',
                          parameters_input,
                          parameters_mllib,
                          parameters_output)
except:
    print('model already exists')
    pass

# inference call
data = [args.img_in]
parameters_input = {
    'rgb':True,
    'scale': 0.00392,
    "mean":[0.5,0.5,0.5],                                                                           
    "std":[0.5,0.5,0.5]
}
parameters_mllib = {
    'extract_layer': 'last'
}
parameters_output = {}
jout = dd.post_predict('testggan',data,
                       parameters_input,
                       parameters_mllib,
                       parameters_output)

#print(jout)
vals = jout['body']['predictions'][0]['vals']
#print('vals=',vals)
np_vals = np.array(vals)
np_vals = np_vals.reshape((3,args.img_size,args.img_size))
out_img = (np.transpose(np_vals, (1, 2, 0)) + 1) / 2.0 * 255.0
out_img = cv2.cvtColor(out_img.astype('float32'), cv2.COLOR_RGB2BGR)
cv2.imwrite(args.img_out,out_img)
print("Successfully generated image " + args.img_out)
