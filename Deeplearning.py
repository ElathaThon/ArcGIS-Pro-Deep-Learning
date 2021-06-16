#Generiques
import os,shutil,sys,time,datetime
from pathlib import Path
import uuid


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

	def __init__(self, project_path):
		self.project_path = project_path
		self.dir_labeled_objects = self.project_path + "/imageChips"
		self.dir_trained_models = self.project_path + "/models"

	def __str__ (self):
		return "+-\n| project_path => {}\n| dir_labeled_objects => {}\n| dir_trained_models => {}\n+-".format(self.project_path, self.dir_labeled_objects, self.dir_trained_models)


class ExportLabelObjects:
	#https://pro.arcgis.com/en/pro-app/latest/tool-reference/image-analyst/export-training-data-for-deep-learning.htm

	deeplearningProject = None

	inRaster = "c:/test/InputRaster.tif" #La imatge o rastermosaic de entrada
	FC_training_data = "c:/test/TrainingData.shp" #FC amb els poligons etiquetats

	image_chip_format = "TIFF"
	tile_size = 64
	stride = 32 #sempre faig la mitat, el calculo diractament en el constructor

	#metadata_format= "Labeled_Tiles" Ho defineixo directament dins de cada una de les funcions de export

	output_nofeature_tiles= "ONLY_TILES_WITH_FEATURES"

	#Default, pero ja estan OK
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

		print("+\n| > Exportem PASCAL({}) => tileSize: {}x{} / stride: {}x{}".format(self.image_chip_format, self.tile_size,self.tile_size, self.stride,self.stride))
		print("| Exporting to... => {}".format(out_folder))
		self.export(self.tile_size, self.stride, metadata_format, self.image_chip_format, out_folder)

	def exportKitti(self):
		out_folder = self.deeplearningProject.dir_labeled_objects + "/KITTI_T" + str(self.tile_size) + "_" + self.image_chip_format + "_" + self.raster_name + "_" + self.randomNumeber()
		metadata_format = "KITTI_rectangles"

		print("+\n| > Exportem KITTI({}) => tileSize: {}x{} / stride: {}x{}".format(self.image_chip_format, self.tile_size,self.tile_size, self.stride,self.stride))
		print("| Exporting to... => {}".format(out_folder))
		self.export(self.tile_size, self.stride, metadata_format, self.image_chip_format, out_folder)

	def export(self, tile_size, stride, metadata_format, img_format, out_folder):

		timer = Timer()

		ExportTrainingDataForDeepLearning(self.inRaster, out_folder, self.FC_training_data, self.image_chip_format,tile_size, tile_size, stride, stride, self.output_nofeature_tiles, metadata_format, self.start_index, self.classvalue_field, self.buffer_radius, self.in_mask_polygons, self.rotation_angle, self.reference_system, self.processing_mode)

		print('| >> Done! - Ha trigat', timer.stop())


	def randomNumeber(self):
		return "" + uuid.uuid4().hex[:8]

	def createDirectory(self, path):
		if not os.path.exists(path):
			print ("Created directory in: " + path)
			os.makedirs(path)




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
