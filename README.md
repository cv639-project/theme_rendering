# Multi-instance theme rendering with neural network
## usage



Download [coco2017 dataset](http://images.cocodataset.org/zips/train2017.zip) and unzip as coco-2017 under root directory.
- Put images to be filtered in photos directory.
- Filtered images will be saved to an automatically created output directory
- **python nn_filters.py --load-model=None** to train from scratch.
- **python nn_filters.py --load-model="path to saved model"** to load a pretrained model.
- Pastiche is the model for theme rendering.
- Feature_extractor is the neural network to extract features.
- Functions for instance segmentation and multi-filtering are included in utils.
- Please refer to our argparse flags for more optional arguments and details.

