# Import system modules
import arcpy
from arcpy.ia import *

# Check out the ArcGIS Image Analyst extension license
arcpy.CheckOutExtension("ImageAnalyst")

#Define input parameters
in_folder = r"C:\Users\Elatha\Desktop\Projecte_GeoAI\ImageChips"
out_folder = r"C:\Users\Elatha\Desktop\Projecte_GeoAI\Models"
max_epochs = 100
model_type = "SSD"
batch_size = 2
arg = "grids '[4, 2, 1]';zooms '[0.7, 1.0, 1.3]';ratios '[[1, 1], [1, 0.5], [0.5, 1]]'"
learning_rate = "" #0.0003
backbone_model = "RESNET34"
pretrained_model = "" #"C:\\Models\\Pretrained\\vehicles.emd"
validation_percent = 30
stop_training = "STOP_TRAINING"
freeze = "UNFREEZE_MODEL" #Descheck dona millor resultats pero triga més


print("Start training: ")
# Execute
TrainDeepLearningModel(in_folder, out_folder, max_epochs, model_type,
     batch_size, arg, learning_rate, backbone_model, pretrained_model,
     validation_percent, stop_training, freeze)

print("FIN!!")
