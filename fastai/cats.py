from fastai.vision.all import *
from fastai.callback.tensorboard import *

print("Downloading dataset...")
path = untar_data(URLs.PETS)/'images'
print("Finished downloading dataset")

def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))

print("On the login node, this will download resnet34")
learn = vision_learner(dls, resnet34, metrics=accuracy)
learn.unfreeze()
learn.fit_one_cycle(3, cbs=TensorBoardCallback('runs', trace_model=True))
