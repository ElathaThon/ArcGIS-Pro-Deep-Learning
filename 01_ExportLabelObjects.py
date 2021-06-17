from Deeplearning import *

project_path = r"E:\Feina_ProjecteGeoAI"

#Imagte/rastermosaic a exportar els labeled objects
#inRaster = r"G:\Projecte_GeoAI\ArcGIS\Rastermosaics.gdb\platges2017"
#inRaster = "G:/Projecte_GeoAI/ArcGIS/imatges/cortadas/Barcelona_2015_cortada/OrtoPlatges_Barcelona_Ago2015_0003_0009.tif"
inRaster = r"G:\Projecte_GeoAI\ArcGIS\Rastermosaics.gdb\platges"

#FC de la GBD o SHP amb les dades ja etiquetades
#FC_training_data = r"G:\Projecte_GeoAI\ArcGIS\Dades_cvc_cuinades.gdb\LabeledData_2017"
FC_training_data = r"G:\Projecte_GeoAI\ArcGIS\Dades_cvc_cuinades.gdb\LabeledData_"

#Format de sortida de les images que es generen
image_chip_format = "TIFF"


project = DeeplearningProject(project_path)
print(project)

tile_sizes = [256,128,64,32]
years = ["2012","2013","2015","2016","2017"]
formats = ["TIFF","JPEG"]

timer = Timer()
for format in formats:
    for tile_size in tile_sizes:
        for year in years:
            raster_entrada = inRaster + year
            dades_cuinades = FC_training_data + year
            export_labeled = ExportLabelObjects(project, raster_entrada, dades_cuinades, format, tile_size)
            export_labeled.exportPascal()
            export_labeled.exportKitti()


print("FIN!!! => Total:", timer.stop())

input("--> Press enter to exit <--")
