# encoding: utf-8
from Deeplearning import *

project_path = r"E:\Feina_ProjecteGeoAI"


project = DeeplearningProject(project_path)
project.log(project)

backbone_models = ["DENSENET201", "DENSENET121", "RESNET152", "VGG19"]


#model_types = {{"model_type": "SSD", "metadata_format": "PASCAL"},{"model_type": "RETINANET", "metadata_format": "PASCAL"}}


folder_labeled_elements = "E:/Feina_ProjecteGeoAI/imageChips/"
pretrained_model = "" #r"E:\Feina_ProjecteGeoAI\models\SSD_DENSENET201_T256_Ep1_v0_1fc08f3a\SSD_DENSENET201_T256_Ep1_v0.emd"

#elements_a_entrenar = ["PASCAL_T256_TIFF_platges2017_5ffe863d","PASCAL_T128_TIFF_platges2017_6e967bb1","PASCAL_T64_TIFF_platges2017_3921d523","PASCAL_T32_TIFF_platges2017_46416e32"]
elements_a_entrenar = ["PASCAL_T256_TIFF_platges2017_5ffe863d","PASCAL_T128_TIFF_platges2017_6e967bb1","PASCAL_T64_TIFF_platges2017_3921d523","PASCAL_T32_TIFF_platges2017_46416e32"]


max_epochs = 80
learning_rate = "" #0.0003

project.log("\n| Start training: ")
timer = Timer()

model_types = ["SSD","RETINANET"]

for labeled_data in elements_a_entrenar:

    in_folder = folder_labeled_elements + labeled_data

    for model_type in model_types:
        for network in backbone_models:
            project.log("| {} - {} => {}".format(model_type, network, in_folder))
            trainer = TrainModel(project, max_epochs, model_type, learning_rate, pretrained_model, network, in_folder)
            trainer.train()





str_fin = "|\n FIN!!! => Total: {}\n+".format(timer.stop())
project.log(str_fin)

project.log("\n+-------------------------------------------------------+\n\n")

input("--> Press enter to exit <--")
