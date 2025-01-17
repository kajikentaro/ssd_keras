from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
import tensorflow as tf
from ssd_v2 import SSD300v2
from ssd_utils import BBoxUtility
np.set_printoptions(suppress=True)
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
set_session(tf.Session(config=config))
voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
               'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
               'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
               'Sheep', 'Sofa', 'Train', 'Tvmonitor']
NUM_CLASSES = len(voc_classes) + 1
input_shape = (300, 300, 3)
model = SSD300v2(input_shape, num_classes=NUM_CLASSES)
model.load_weights('weights_SSD300.hdf5', by_name=True)
bbox_util = BBoxUtility(NUM_CLASSES)

def pltToCV2(fig):
	import io
	buf = io.BytesIO()  # インメモリのバイナリストリームを作成
	fig.savefig(buf, format="png", dpi=180)  # matplotlibから出力される画像のバイナリデータをメモリに格納する.
	buf.seek(0)  # ストリーム位置を先頭に戻る
	img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)  # メモリからバイナリデータを読み込み, numpy array 形式に変換
	buf.close()  # ストリームを閉じる(flushする)
	img = cv2.imdecode(img_arr, 1)  # 画像のバイナリデータを復元する
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2.imread() はBGR形式で読み込むのでRGBにする.
	return img

def predict(inputs, images):
	preds = model.predict(inputs, batch_size=1, verbose=1)
	results = bbox_util.detection_out(preds)
	for i, img in enumerate(images):
		# Parse the outputs.
		det_label = results[i][:, 0]
		det_conf = results[i][:, 1]
		det_xmin = results[i][:, 2]
		det_ymin = results[i][:, 3]
		det_xmax = results[i][:, 4]
		det_ymax = results[i][:, 5]

		# Get detections with confidence higher than 0.6.
		top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

		top_conf = det_conf[top_indices]
		top_label_indices = det_label[top_indices].tolist()
		top_xmin = det_xmin[top_indices]
		top_ymin = det_ymin[top_indices]
		top_xmax = det_xmax[top_indices]
		top_ymax = det_ymax[top_indices]

		colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
		plt.imshow(img / 255.)
		currentAxis = plt.gca()

		for i in range(top_conf.shape[0]):
			xmin = int(round(top_xmin[i] * img.shape[1]))
			ymin = int(round(top_ymin[i] * img.shape[0]))
			xmax = int(round(top_xmax[i] * img.shape[1]))
			ymax = int(round(top_ymax[i] * img.shape[0]))
			score = top_conf[i]
			label = int(top_label_indices[i])
			label_name = voc_classes[label - 1]
			display_txt = '{:0.2f}, {}'.format(score, label_name)
			coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
			color = colors[label]
			currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
			currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})
		return plt

import cv2
import time
from keras.preprocessing import image
import numpy as np
cap = cv2.VideoCapture(0)
while True:
	#time.sleep(1)
	inputs = []
	images = []

	ret, frame = cap.read()
	images.append(frame)
	frame = image.img_to_array(frame)

	resized = cv2.resize(frame, (300, 300))
	inputs.append(resized)

	inputs = preprocess_input(np.array(inputs))
	plt = predict(inputs, images)
	img = pltToCV2(plt)
	plt.close()

	cv2.imshow('tmp',img)

	cv2.waitKey(1)



cv2.destroyAllWindows()
cap.release()