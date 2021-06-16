from Deeplearning import *

project_path = "G:/Projecte_GeoAI/ArcGIS"

#Imagte/rastermosaic a exportar els labeled objects
#inRaster = "G:/Projecte_GeoAI/ArcGIS/GeoAI platges/Projecte_GeoAI/scratch.gdb/platges2015"
inRaster = "G:/Projecte_GeoAI/ArcGIS/imatges/cortadas/Barcelona_2015_cortada/OrtoPlatges_Barcelona_Ago2015_0003_0009.tif"

#FC de la GBD o SHP amb les dades ja etiquetades
FC_training_data = r"G:\Projecte_GeoAI\ArcGIS\Dades_cvc_cuinades.gdb\LabeledData_2015"

#Format de sortida de les images que es generen
image_chip_format = "TIFF"


project = DeeplearningProject(project_path)
print(project)

tile_sizes = [256,128,64,32]

timer = Timer()
for tile_size in tile_sizes:
    export_labeled = ExportLabelObjects(project, inRaster, FC_training_data, image_chip_format, tile_size)
    export_labeled.exportPascal()
    export_labeled.exportKitti()


print("FIN!!! => Total:", timer.stop())

#input("Press enter to exit ;)")
