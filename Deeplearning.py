# encoding: utf-8
#Generiques
import os,shutil,sys,time,datetime
from pathlib import Path
import uuid

import logging

#Les de ArcGIS
import arcpy
arcpy.CheckOutExtension("ImageAnalyst")
from arcpy.ia import *

arcpy.env.overwriteOutput =True



class DeeplearningProject:
	""" Definim les variables generiques del projecte d'ArcGIS """

	project_path = ""
	dir_labeled_objects = ""
	dir_trained_models  = ""

	loggery = None

	def __init__(self, project_path):
		self.project_path = project_path
		self.dir_labeled_objects = self.project_path + "/imageChips"
		self.dir_trained_models = self.project_path + "/models"

		self.loggery = Log(project_path)

	def __str__ (self):
		return "+------\n| project_path => {}\n| dir_labeled_objects => {}\n| dir_trained_models => {}\n+------".format(self.project_path, self.dir_labeled_objects, self.dir_trained_models)

	def log(self,string):
		self.loggery.log(string)


class ExportLabelObjects:
	#https://pro.arcgis.com/en/pro-app/latest/tool-reference/image-analyst/export-training-data-for-deep-learning.htm

	deeplearningProject = None

	inRaster = "c:/test/InputRaster.tif" #La imatge o rastermosaic de entrada
	FC_training_data = "c:/test/TrainingData.shp" #FC amb els poligons etiquetats
	image_chip_format = "TIFF"
	tile_size = 64
	stride = 32 #sempre faig la mitat, el calculo diractament en el constructor

	#metadata_format= "Labeled_Tiles" Ho defineixo directament dins de cada una de les funcions de export


	#Default, pero ja estan OK
	output_nofeature_tiles= "ONLY_TILES_WITH_FEATURES"
	start_index = 0
	classvalue_field = "Classvalue"
	buffer_radius = 0
	in_mask_polygons = ""
	rotation_angle = 0
	reference_system = "PIXEL_SPACE"
	processing_mode = "PROCESS_AS_MOSAICKED_IMAGE"


	#Per coses de la Class
	raster_name = ""

	def __init__(self, deeplearningProject, inRaster, FC_training_data, image_chip_format, tile_size):
		self.deeplearningProject = deeplearningProject
		self.inRaster = inRaster
		self.FC_training_data = FC_training_data
		self.image_chip_format = image_chip_format
		self.tile_size = tile_size
		self.stride = self.tile_size/2
		self.raster_name = self.nomImatge()

	def nomImatge(self):
		return os.path.splitext(os.path.basename(self.inRaster))[0]


	def exportPascal(self):
		out_folder = self.deeplearningProject.dir_labeled_objects + "/PASCAL_T" + str(self.tile_size) + "_" + self.image_chip_format + "_" + self.raster_name + "_" + self.randomNumeber()
		metadata_format = "PASCAL_VOC_rectangles"

		self.deeplearningProject.log("+\n| > Exportem PASCAL({}) => tileSize: {}x{} / stride: {}x{}".format(self.image_chip_format, self.tile_size,self.tile_size, self.stride,self.stride))
		self.deeplearningProject.log("| Exporting to... => {}".format(out_folder))
		self.export(self.tile_size, self.stride, metadata_format, self.image_chip_format, out_folder)

	def exportKitti(self):
		out_folder = self.deeplearningProject.dir_labeled_objects + "/KITTI_T" + str(self.tile_size) + "_" + self.image_chip_format + "_" + self.raster_name + "_" + self.randomNumeber()
		metadata_format = "KITTI_rectangles"

		self.deeplearningProject.log("+\n| > Exportem KITTI({}) => tileSize: {}x{} / stride: {}x{}".format(self.image_chip_format, self.tile_size,self.tile_size, self.stride,self.stride))
		self.deeplearningProject.log("| Exporting to... => {}".format(out_folder))
		self.export(self.tile_size, self.stride, metadata_format, self.image_chip_format, out_folder)

	def export(self, tile_size, stride, metadata_format, img_format, out_folder):

		timer = Timer()

		ExportTrainingDataForDeepLearning(self.inRaster, out_folder, self.FC_training_data, self.image_chip_format,tile_size, tile_size, stride, stride, self.output_nofeature_tiles, metadata_format, self.start_index, self.classvalue_field, self.buffer_radius, self.in_mask_polygons, self.rotation_angle, self.reference_system, self.processing_mode)

		self.deeplearningProject.log('| >> Done! - Ha trigat', timer.stop())


	def randomNumeber(self):
		return "" + uuid.uuid4().hex[:8]



class TrainModel:
	#https://pro.arcgis.com/en/pro-app/2.6/tool-reference/image-analyst/train-deep-learning-model.htm

	deeplearningProject = None

	#Per jugar amb elles
	in_folder = r"C:\Users\Elatha\Desktop\Projecte_GeoAI\ImageChips"
	max_epochs = 100
	model_type = "SSD"
	learning_rate = "" #0.0003
	pretrained_model = "" #"C:\\Models\\Pretrained\\vehicles.emd"
	backbone_model = "RESNET34"


	# Per defecte OK
	freeze = "UNFREEZE_MODEL" #Descheck dona millor resultats pero triga més
	validation_percent = 40
	batch_size = 10 #Les imatges que es fan a l'hora contra més alt millor GPU s'ha de tenir. default = 2
	arg = "grids '[4, 2, 1]';zooms '[0.7, 1.0, 1.3]';ratios '[[1, 1], [1, 0.5], [0.5, 1]]'"
	stop_training = "STOP_TRAINING"

	#Les definim amb el projecte
	out_folder = r"C:\Users\Elatha\Desktop\Projecte_GeoAI\Models"

	#Per coses de la Class
	out_model_name = ""
	train_version = 0
	platgesQueFem = ""


	def __init__(self,deeplearningProject,max_epochs,model_type,learning_rate,pretrained_model,backbone_model,in_folder):
		self.deeplearningProject = deeplearningProject
		self.max_epochs = max_epochs
		self.model_type = model_type
		self.learning_rate = learning_rate
		self.pretrained_model = pretrained_model
		self.backbone_model = backbone_model
		self.in_folder = in_folder
		self.out_model_name = self.model_name()

		if self.havePreviousTrainedModel():
			self.train_version = self.getVersionNumber()


	def havePreviousTrainedModel(self):
		if self.pretrained_model != "":
			return True
		else:
			return False

	def getVersionNumber(self):
		versionNumber = os.path.splitext(os.path.basename(self.pretrained_model))[0].split(sep="_")[4].replace("v","")
		actual_version = int(versionNumber)
		next_version = actual_version + 1
		#print("Ara: {} => toca la {}".format(actual_version,next_version))
		return next_version

	def model_name(self):
		labeled_data_info = os.path.splitext(os.path.basename(self.in_folder))[0].split(sep="_")
		self.platgesQueFem = labeled_data_info[3]
		#print(labeled_data_info)
		model_name = self.model_type + "_" + self.backbone_model + "_" + labeled_data_info[1] + "_" + self.platgesQueFem + "_v" + str(self.train_version) + "_Ep" + str(self.max_epochs)

		return model_name


	def train(self):
		out_folder = self.deeplearningProject.dir_trained_models + "/" + self.out_model_name

		timer = Timer()
		self.deeplearningProject.log("| > Epocas: {}\n| > AI Network: {}".format(self.max_epochs, self.backbone_model))
		learner = TrainDeepLearningModel(self.in_folder, out_folder, self.max_epochs, self.model_type, self.batch_size, self.arg, self.learning_rate, self.backbone_model, self.pretrained_model, self.validation_percent, self.stop_training, self.freeze)

		self.deeplearningProject.log("| >> Done! - Ha trigat {}\n+-".format(timer.stop()))

	def randomNumeber(self):
		return "" + uuid.uuid4().hex[:8]



class Timer:

	startTime = None
	endTime = None
	diff = 0

	def __init__(self):
		self.start()

	def start(self):
		self.startTime = time.time()

	def getCurrentTime(self):
		return time.time()

	def stop(self):
		self.endTime = self.getCurrentTime()
		self.diff()
		return self.__str__()

	def diff(self):
		self.diff = int(self.endTime - self.startTime)

	def __str__(self):
		return str(datetime.timedelta(seconds = self.diff))


class Log:

	def __init__(self, path):
		"""
		self.createDirectory(path)
		fileh = logging.FileHandler(path, 'a')
		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		fileh.setFormatter(formatter)

		log = logging.getLogger()  # root logger
		for hdlr in log.handlers[:]:  # remove all old handlers
		    log.removeHandler(hdlr)
		log.addHandler(fileh)      # set the new handler
		"""
		#logging.FileHandler(path, 'a')
		logging.basicConfig(filename="logfilename.log", level=logging.INFO, format='%(message)s')

	def log(self, string):
		logging.info(string)
		print(string)


	def createDirectory(self, path):
		if not os.path.exists(path):
			print ("Created directory in: " + path)
			os.makedirs(path)
