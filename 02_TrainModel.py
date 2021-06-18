from Deeplearning import *

project_path = r"E:\Feina_ProjecteGeoAI"


project = DeeplearningProject(project_path)
print(project)

backbone_models = ["DENSENET201", "DENSENET121", "RESNET152", "VGG19"]

#model_types = {{"model_type": "SSD", "metadata_format": "PASCAL"},{"model_type": "RETINANET", "metadata_format": "PASCAL"}}


in_folder = r"E:\Feina_ProjecteGeoAI\imageChips\PASCAL_T256_TIFF_platges2017_5ffe863d"
pretrained_model = "" #r"E:\Feina_ProjecteGeoAI\models\SSD_DENSENET201_T256_Ep1_v0_1fc08f3a\SSD_DENSENET201_T256_Ep1_v0.emd"


max_epochs = 50
learning_rate = "" #0.0003

print("+-\n| Start training: ")
timer = Timer()

model_type = "SSD"
backbone_model = "DENSENET201"

trainer = TrainModel(project, max_epochs, model_type, learning_rate, pretrained_model, backbone_model, in_folder)

trainer.train()


print("FIN!!! => Total:", timer.stop())

input("--> Press enter to exit <--")
