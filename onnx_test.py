import json
import pickle
import os
import matplotlib
import numpy as np
import onnx
import onnxruntime
import cv2
from matplotlib import pyplot as plt

model = "./trainednet.onnx"
dataset_dir = "./output/landsat_datasets_manual_5bands/"
n1 = onnx.helper.make_node('Sigmoid', inputs=['x'], outputs=['preds'], name='n1')
g1 = onnx.helper.make_graph([n1], 'sigmoid',
                            [onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, shape=(161, 105))],
                            [onnx.helper.make_tensor_value_info('preds', onnx.TensorProto.FLOAT, shape=(161, 105))])
m1 = onnx.helper.make_model(g1, producer_name='ben')

onnx.save(m1, 'sigmoid.onnx')
xd = cv2.dnn.readNetFromONNX('./trainednet.onnx')
full_dataset = []
for str in os.listdir(dataset_dir):
    f = open(dataset_dir + str, "rb")
    dataset = pickle.load(f)
    f.close()
    full_dataset.extend(dataset)
img = full_dataset[0]
img = img[0].numpy()
img = np.expand_dims(img, axis=0)
data = json.dumps({'data': img.tolist()})
data = np.array(json.loads(data)['data']).astype('float32')
session = onnxruntime.InferenceSession(model, None)
sigmoid_session = onnxruntime.InferenceSession('sigmoid.onnx')
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
result = session.run([output_name], {input_name: data})
probs = sigmoid_session.run([], {'x': result[0][0][1]})
threshold = 0.666
probs = probs[0]
pred = (probs > threshold).astype(float)
matplotlib.use('TkAgg')
# plt.imshow(img[0][0])
# plt.show()
plt.imshow(pred)
plt.savefig("pred_result.png")
# plt.show()
