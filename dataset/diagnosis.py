import os
import csv
import glob
import warnings
import numpy as np
import pandas as pd
import torch
from PIL import Image
from os.path import expanduser
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


def get_default_root():
    home = expanduser("~")
    dirpath = os.path.join(home, ".medmnist")

    try:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
    except:
        warnings.warn("Failed to setup default root.")
        dirpath = None

    return dirpath


DEFAULT_ROOT = get_default_root()

HOMEPAGE = "https://github.com/MedMNIST/MedMNIST/"

INFO = {
    "pathmnist": {
        "python_class": "PathMNIST",
        "description": "The PathMNIST is based on a prior study for predicting survival from colorectal cancer histology slides, providing a dataset (NCT-CRC-HE-100K) of 100,000 non-overlapping image patches from hematoxylin & eosin stained histological images, and a test dataset (CRC-VAL-HE-7K) of 7,180 image patches from a different clinical center. The dataset is comprised of 9 types of tissues, resulting in a multi-class classification task. We resize the source images of 3×224×224 into 3×28×28, and split NCT-CRC-HE-100K into training and validation set with a ratio of 9:1. The CRC-VAL-HE-7K is treated as the test set.",
        "url": "https://zenodo.org/records/10519652/files/pathmnist.npz?download=1",
        "MD5": "a8b06965200029087d5bd730944a56c1",
        "url_64": "https://zenodo.org/records/10519652/files/pathmnist_64.npz?download=1",
        "MD5_64": "55aa9c1e0525abe5a6b9d8343a507616",
        "url_128": "https://zenodo.org/records/10519652/files/pathmnist_128.npz?download=1",
        "MD5_128": "ac42d08fb904d92c244187169d1fd1d9",
        "url_224": "https://zenodo.org/records/10519652/files/pathmnist_224.npz?download=1",
        "MD5_224": "2c51a510bcdc9cf8ddb2af93af1eadec",
        "task": "multi-class",
        "label": {
            "0": "adipose",
            "1": "background",
            "2": "debris",
            "3": "lymphocytes",
            "4": "mucus",
            "5": "smooth muscle",
            "6": "normal colon mucosa",
            "7": "cancer-associated stroma",
            "8": "colorectal adenocarcinoma epithelium",
        },
        "n_channels": 3,
        "n_samples": {"train": 89996, "val": 10004, "test": 7180},
        "license": "CC BY 4.0",
    },
    "chestmnist": {
        "python_class": "ChestMNIST",
        "description": "The ChestMNIST is based on the NIH-ChestXray14 dataset, a dataset comprising 112,120 frontal-view X-Ray images of 30,805 unique patients with the text-mined 14 disease labels, which could be formulized as a multi-label binary-class classification task. We use the official data split, and resize the source images of 1×1024×1024 into 1×28×28.",
        "url": "https://zenodo.org/records/10519652/files/chestmnist.npz?download=1",
        "MD5": "02c8a6516a18b556561a56cbdd36c4a8",
        "url_64": "https://zenodo.org/records/10519652/files/chestmnist_64.npz?download=1",
        "MD5_64": "9de6cd0b934ebb5b7426cfba5efbae16",
        "url_128": "https://zenodo.org/records/10519652/files/chestmnist_128.npz?download=1",
        "MD5_128": "db107e5590b27930b62dbaf558aebee3",
        "url_224": "https://zenodo.org/records/10519652/files/chestmnist_224.npz?download=1",
        "MD5_224": "45bd33e6f06c3e8cdb481c74a89152aa",
        "task": "multi-label, binary-class",
        "label": {
            "0": "atelectasis",
            "1": "cardiomegaly",
            "2": "effusion",
            "3": "infiltration",
            "4": "mass",
            "5": "nodule",
            "6": "pneumonia",
            "7": "pneumothorax",
            "8": "consolidation",
            "9": "edema",
            "10": "emphysema",
            "11": "fibrosis",
            "12": "pleural",
            "13": "hernia",
        },
        "n_channels": 1,
        "n_samples": {"train": 78468, "val": 11219, "test": 22433},
        "license": "CC BY 4.0",
    },
    "dermamnist": {
        "python_class": "DermaMNIST",
        "description": "The DermaMNIST is based on the HAM10000, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. The dataset consists of 10,015 dermatoscopic images categorized as 7 different diseases, formulized as a multi-class classification task. We split the images into training, validation and test set with a ratio of 7:1:2. The source images of 3×600×450 are resized into 3×28×28.",
        "url": "https://zenodo.org/records/10519652/files/dermamnist.npz?download=1",
        "MD5": "0744692d530f8e62ec473284d019b0c7",
        "url_64": "https://zenodo.org/records/10519652/files/dermamnist_64.npz?download=1",
        "MD5_64": "b70a2f5635c6199aeaa28c31d7202e1f",
        "url_128": "https://zenodo.org/records/10519652/files/dermamnist_128.npz?download=1",
        "MD5_128": "2defd784463fa5243564e855ed717de1",
        "url_224": "https://zenodo.org/records/10519652/files/dermamnist_224.npz?download=1",
        "MD5_224": "8974907d8e169bef5f5b96bc506ae45d",
        "task": "multi-class",
        "label": {
            "0": "actinic keratoses and intraepithelial carcinoma",
            "1": "basal cell carcinoma",
            "2": "benign keratosis-like lesions",
            "3": "dermatofibroma",
            "4": "melanoma",
            "5": "melanocytic nevi",
            "6": "vascular lesions",
        },
        "n_channels": 3,
        "n_samples": {"train": 7007, "val": 1003, "test": 2005},
        "license": "CC BY-NC 4.0",
    },
    "octmnist": {
        "python_class": "OCTMNIST",
        "description": "The OCTMNIST is based on a prior dataset of 109,309 valid optical coherence tomography (OCT) images for retinal diseases. The dataset is comprised of 4 diagnosis categories, leading to a multi-class classification task. We split the source training set with a ratio of 9:1 into training and validation set, and use its source validation set as the test set. The source images are gray-scale, and their sizes are (384−1,536)×(277−512). We center-crop the images and resize them into 1×28×28.",
        "url": "https://zenodo.org/records/10519652/files/octmnist.npz?download=1",
        "MD5": "c68d92d5b585d8d81f7112f81e2d0842",
        "url_64": "https://zenodo.org/records/10519652/files/octmnist_64.npz?download=1",
        "MD5_64": "e229e9440236b774d9f0dfef9d07bdaf",
        "url_128": "https://zenodo.org/records/10519652/files/octmnist_128.npz?download=1",
        "MD5_128": "0a97e76651ace45c5d943ee3f65b63ae",
        "url_224": "https://zenodo.org/records/10519652/files/octmnist_224.npz?download=1",
        "MD5_224": "abc493b6d529d5de7569faaef2773ba3",
        "task": "multi-class",
        "label": {
            "0": "choroidal neovascularization",
            "1": "diabetic macular edema",
            "2": "drusen",
            "3": "normal",
        },
        "n_channels": 1,
        "n_samples": {"train": 97477, "val": 10832, "test": 1000},
        "license": "CC BY 4.0",
    },
    "pneumoniamnist": {
        "python_class": "PneumoniaMNIST",
        "description": "The PneumoniaMNIST is based on a prior dataset of 5,856 pediatric chest X-Ray images. The task is binary-class classification of pneumonia against normal. We split the source training set with a ratio of 9:1 into training and validation set and use its source validation set as the test set. The source images are gray-scale, and their sizes are (384−2,916)×(127−2,713). We center-crop the images and resize them into 1×28×28.",
        "url": "https://zenodo.org/records/10519652/files/pneumoniamnist.npz?download=1",
        "MD5": "28209eda62fecd6e6a2d98b1501bb15f",
        "url_64": "https://zenodo.org/records/10519652/files/pneumoniamnist_64.npz?download=1",
        "MD5_64": "8f4eceb4ccffa70c672198ea285246c6",
        "url_128": "https://zenodo.org/records/10519652/files/pneumoniamnist_128.npz?download=1",
        "MD5_128": "05b46931834c231683c68f40c47b2971",
        "url_224": "https://zenodo.org/records/10519652/files/pneumoniamnist_224.npz?download=1",
        "MD5_224": "d6a3c71de1b945ea11211b03746c1fe1",
        "task": "binary-class",
        "label": {"0": "normal", "1": "pneumonia"},
        "n_channels": 1,
        "n_samples": {"train": 4708, "val": 524, "test": 624},
        "license": "CC BY 4.0",
    },
    "retinamnist": {
        "python_class": "RetinaMNIST",
        "description": "The RetinaMNIST is based on the DeepDRiD challenge, which provides a dataset of 1,600 retina fundus images. The task is ordinal regression for 5-level grading of diabetic retinopathy severity. We split the source training set with a ratio of 9:1 into training and validation set, and use the source validation set as the test set. The source images of 3×1,736×1,824 are center-cropped and resized into 3×28×28.",
        "url": "https://zenodo.org/records/10519652/files/retinamnist.npz?download=1",
        "MD5": "bd4c0672f1bba3e3a89f0e4e876791e4",
        "url_64": "https://zenodo.org/records/10519652/files/retinamnist_64.npz?download=1",
        "MD5_64": "afda852cc34dcda56f86ad2b2457dbcc",
        "url_128": "https://zenodo.org/records/10519652/files/retinamnist_128.npz?download=1",
        "MD5_128": "e48e916a24454daf90583d4e6efb1a18",
        "url_224": "https://zenodo.org/records/10519652/files/retinamnist_224.npz?download=1",
        "MD5_224": "eae7e3b6f3fcbda4ae613ebdcbe35348",
        "task": "ordinal-regression",
        "label": {"0": "0", "1": "1", "2": "2", "3": "3", "4": "4"},
        "n_channels": 3,
        "n_samples": {"train": 1080, "val": 120, "test": 400},
        "license": "CC BY 4.0",
    },
    "breastmnist": {
        "python_class": "BreastMNIST",
        "description": "The BreastMNIST is based on a dataset of 780 breast ultrasound images. It is categorized into 3 classes: normal, benign, and malignant. As we use low-resolution images, we simplify the task into binary classification by combining normal and benign as positive and classifying them against malignant as negative. We split the source dataset with a ratio of 7:1:2 into training, validation and test set. The source images of 1×500×500 are resized into 1×28×28.",
        "url": "https://zenodo.org/records/10519652/files/breastmnist.npz?download=1",
        "MD5": "750601b1f35ba3300ea97c75c52ff8f6",
        "url_64": "https://zenodo.org/records/10519652/files/breastmnist_64.npz?download=1",
        "MD5_64": "742edef2a1fd1524b2efff4bd7ba9364",
        "url_128": "https://zenodo.org/records/10519652/files/breastmnist_128.npz?download=1",
        "MD5_128": "363e4b3f8d712e9b5de15470a2aaadf1",
        "url_224": "https://zenodo.org/records/10519652/files/breastmnist_224.npz?download=1",
        "MD5_224": "b56378a6eefa9fed602bb16d192d4c8b",
        "task": "binary-class",
        "label": {"0": "malignant", "1": "normal, benign"},
        "n_channels": 1,
        "n_samples": {"train": 546, "val": 78, "test": 156},
        "license": "CC BY 4.0",
    },
    "bloodmnist": {
        "python_class": "BloodMNIST",
        "description": "The BloodMNIST is based on a dataset of individual normal cells, captured from individuals without infection, hematologic or oncologic disease and free of any pharmacologic treatment at the moment of blood collection. It contains a total of 17,092 images and is organized into 8 classes. We split the source dataset with a ratio of 7:1:2 into training, validation and test set. The source images with resolution 3×360×363 pixels are center-cropped into 3×200×200, and then resized into 3×28×28.",
        "url": "https://zenodo.org/records/10519652/files/bloodmnist.npz?download=1",
        "MD5": "7053d0359d879ad8a5505303e11de1dc",
        "url_64": "https://zenodo.org/records/10519652/files/bloodmnist_64.npz?download=1",
        "MD5_64": "2b94928a2ae4916078ca51e05b6b800b",
        "url_128": "https://zenodo.org/records/10519652/files/bloodmnist_128.npz?download=1",
        "MD5_128": "adace1e0ed228fccda1f39692059dd4c",
        "url_224": "https://zenodo.org/records/10519652/files/bloodmnist_224.npz?download=1",
        "MD5_224": "b718ff6835fcbdb22ba9eacccd7b2601",
        "task": "multi-class",
        "label": {
            "0": "basophil",
            "1": "eosinophil",
            "2": "erythroblast",
            "3": "immature granulocytes(myelocytes, metamyelocytes and promyelocytes)",
            "4": "lymphocyte",
            "5": "monocyte",
            "6": "neutrophil",
            "7": "platelet",
        },
        "n_channels": 3,
        "n_samples": {"train": 11959, "val": 1712, "test": 3421},
        "license": "CC BY 4.0",
    },
    "tissuemnist": {
        "python_class": "TissueMNIST",
        "description": "We use the BBBC051, available from the Broad Bioimage Benchmark Collection. The dataset contains 236,386 human kidney cortex cells, segmented from 3 reference tissue specimens and organized into 8 categories. We split the source dataset with a ratio of 7:1:2 into training, validation and test set. Each gray-scale image is 32×32×7 pixels, where 7 denotes 7 slices. We take maximum values across the slices and resize them into 28×28 gray-scale images.",
        "url": "https://zenodo.org/records/10519652/files/tissuemnist.npz?download=1",
        "MD5": "ebe78ee8b05294063de985d821c1c34b",
        "url_64": "https://zenodo.org/records/10519652/files/tissuemnist_64.npz?download=1",
        "MD5_64": "123ece2eba09d0aa5d698fda57103344",
        "url_128": "https://zenodo.org/records/10519652/files/tissuemnist_128.npz?download=1",
        "MD5_128": "61b955355d7425a89687b06cca3ce0c2",
        "url_224": "https://zenodo.org/records/10519652/files/tissuemnist_224.npz?download=1",
        "MD5_224": "b077128c4a949f0a4eb01517f9037b9c",
        "task": "multi-class",
        "label": {
            "0": "Collecting Duct, Connecting Tubule",
            "1": "Distal Convoluted Tubule",
            "2": "Glomerular endothelial cells",
            "3": "Interstitial endothelial cells",
            "4": "Leukocytes",
            "5": "Podocytes",
            "6": "Proximal Tubule Segments",
            "7": "Thick Ascending Limb",
        },
        "n_channels": 1,
        "n_samples": {"train": 165466, "val": 23640, "test": 47280},
        "license": "CC BY 4.0",
    },
    "organamnist": {
        "python_class": "OrganAMNIST",
        "description": "The OrganAMNIST is based on 3D computed tomography (CT) images from Liver Tumor Segmentation Benchmark (LiTS). It is renamed from OrganMNIST_Axial (in MedMNIST v1) for simplicity. We use bounding-box annotations of 11 body organs from another study to obtain the organ labels. Hounsfield-Unit (HU) of the 3D images are transformed into gray-scale with an abdominal window. We crop 2D images from the center slices of the 3D bounding boxes in axial views (planes). The images are resized into 1×28×28 to perform multi-class classification of 11 body organs. 115 and 16 CT scans from the source training set are used as training and validation set, respectively. The 70 CT scans from the source test set are treated as the test set.",
        "url": "https://zenodo.org/records/10519652/files/organamnist.npz?download=1",
        "MD5": "68e3f8846a6bd62f0c9bf841c0d9eacc",
        "url_64": "https://zenodo.org/records/10519652/files/organamnist_64.npz?download=1",
        "MD5_64": "2dcccc29b88e6da5a01161ef20cda288",
        "url_128": "https://zenodo.org/records/10519652/files/organamnist_128.npz?download=1",
        "MD5_128": "eeae80d0a227a8d099027e1b3cfd3b60",
        "url_224": "https://zenodo.org/records/10519652/files/organamnist_224.npz?download=1",
        "MD5_224": "50747347e05c87dd3aaf92c49f9f3170",
        "task": "multi-class",
        "label": {
            "0": "bladder",
            "1": "femur-left",
            "2": "femur-right",
            "3": "heart",
            "4": "kidney-left",
            "5": "kidney-right",
            "6": "liver",
            "7": "lung-left",
            "8": "lung-right",
            "9": "pancreas",
            "10": "spleen",
        },
        "n_channels": 1,
        "n_samples": {"train": 34561, "val": 6491, "test": 17778},
        "license": "CC BY 4.0",
    },
    "organcmnist": {
        "python_class": "OrganCMNIST",
        "description": "The OrganCMNIST is based on 3D computed tomography (CT) images from Liver Tumor Segmentation Benchmark (LiTS). It is renamed from OrganMNIST_Coronal (in MedMNIST v1) for simplicity. We use bounding-box annotations of 11 body organs from another study to obtain the organ labels. Hounsfield-Unit (HU) of the 3D images are transformed into gray-scale with an abdominal window. We crop 2D images from the center slices of the 3D bounding boxes in coronal views (planes). The images are resized into 1×28×28 to perform multi-class classification of 11 body organs. 115 and 16 CT scans from the source training set are used as training and validation set, respectively. The 70 CT scans from the source test set are treated as the test set.",
        "url": "https://zenodo.org/records/10519652/files/organcmnist.npz?download=1",
        "MD5": "b9ceb9546e10131b32923c5bbeaea2b1",
        "url_64": "https://zenodo.org/records/10519652/files/organcmnist_64.npz?download=1",
        "MD5_64": "3ce34a8724ea6f548e6db4744d03b6a9",
        "url_128": "https://zenodo.org/records/10519652/files/organcmnist_128.npz?download=1",
        "MD5_128": "773c1f009daa3fe5d9a2a201b2a7ed94",
        "url_224": "https://zenodo.org/records/10519652/files/organcmnist_224.npz?download=1",
        "MD5_224": "050f5e875dc056f6768abf94ec9995d1",
        "task": "multi-class",
        "label": {
            "0": "bladder",
            "1": "femur-left",
            "2": "femur-right",
            "3": "heart",
            "4": "kidney-left",
            "5": "kidney-right",
            "6": "liver",
            "7": "lung-left",
            "8": "lung-right",
            "9": "pancreas",
            "10": "spleen",
        },
        "n_channels": 1,
        "n_samples": {"train": 12975, "val": 2392, "test": 8216},
        "license": "CC BY 4.0",
    },
    "organsmnist": {
        "python_class": "OrganSMNIST",
        "description": "The OrganSMNIST is based on 3D computed tomography (CT) images from Liver Tumor Segmentation Benchmark (LiTS). It is renamed from OrganMNIST_Sagittal (in MedMNIST v1) for simplicity. We use bounding-box annotations of 11 body organs from another study to obtain the organ labels. Hounsfield-Unit (HU) of the 3D images are transformed into gray-scale with an abdominal window. We crop 2D images from the center slices of the 3D bounding boxes in sagittal views (planes). The images are resized into 1×28×28 to perform multi-class classification of 11 body organs. 115 and 16 CT scans from the source training set are used as training and validation set, respectively. The 70 CT scans from the source test set are treated as the test set.",
        "url": "https://zenodo.org/records/10519652/files/organsmnist.npz?download=1",
        "MD5": "9ab87b696fb54e2a387ebe992d6ed5f1",
        "url_64": "https://zenodo.org/records/10519652/files/organsmnist_64.npz?download=1",
        "MD5_64": "53a6d115339d874c25e309a994ff46d3",
        "url_128": "https://zenodo.org/records/10519652/files/organsmnist_128.npz?download=1",
        "MD5_128": "ded0c5fa01a95dc4978b956f613e9b8e",
        "url_224": "https://zenodo.org/records/10519652/files/organsmnist_224.npz?download=1",
        "MD5_224": "b354719e553fbbb2513d5533f52a4cb1",
        "task": "multi-class",
        "label": {
            "0": "bladder",
            "1": "femur-left",
            "2": "femur-right",
            "3": "heart",
            "4": "kidney-left",
            "5": "kidney-right",
            "6": "liver",
            "7": "lung-left",
            "8": "lung-right",
            "9": "pancreas",
            "10": "spleen",
        },
        "n_channels": 1,
        "n_samples": {"train": 13932, "val": 2452, "test": 8827},
        "license": "CC BY 4.0",
    },
    "organmnist3d": {
        "python_class": "OrganMNIST3D",
        "description": "The source of the OrganMNIST3D is the same as that of the Organ{A,C,S}MNIST. Instead of 2D images, we directly use the 3D bounding boxes and process the images into 28×28×28 to perform multi-class classification of 11 body organs. The same 115 and 16 CT scans as the Organ{A,C,S}MNIST from the source training set are used as training and validation set, respectively, and the same 70 CT scans as the Organ{A,C,S}MNIST from the source test set are treated as the test set.",
        "url": "https://zenodo.org/records/10519652/files/organmnist3d.npz?download=1",
        "MD5": "a0c5a1ff56af4f155c46d46fbb45a2fe",
        "url_64": "https://zenodo.org/records/10519652/files/organmnist3d_64.npz?download=1",
        "MD5_64": "58a2205adf14a9d0a189cb06dc78bf10",
        "task": "multi-class",
        "label": {
            "0": "liver",
            "1": "kidney-right",
            "2": "kidney-left",
            "3": "femur-right",
            "4": "femur-left",
            "5": "bladder",
            "6": "heart",
            "7": "lung-right",
            "8": "lung-left",
            "9": "spleen",
            "10": "pancreas",
        },
        "n_channels": 1,
        "n_samples": {"train": 971, "val": 161, "test": 610},
        "license": "CC BY 4.0",
    },
    "nodulemnist3d": {
        "python_class": "NoduleMNIST3D",
        "description": "The NoduleMNIST3D is based on the LIDC-IDRI, a large public lung nodule dataset, containing images from thoracic CT scans. The dataset is designed for both lung nodule segmentation and 5-level malignancy classification task. To perform binary classification, we categorize cases with malignancy level 1/2 into negative class and 4/5 into positive class, ignoring the cases with malignancy level 3. We split the source dataset with a ratio of 7:1:2 into training, validation and test set, and center-crop the spatially normalized images (with a spacing of 1mm×1mm×1mm) into 28×28×28.",
        "url": "https://zenodo.org/records/10519652/files/nodulemnist3d.npz?download=1",
        "MD5": "8755a7e9e05a4d9ce80a24c3e7a256f3",
        "url_64": "https://zenodo.org/records/10519652/files/nodulemnist3d_64.npz?download=1",
        "MD5_64": "c47c5b7d457bf6332200d2ea6d64ecd8",
        "task": "binary-class",
        "label": {"0": "benign", "1": "malignant"},
        "n_channels": 1,
        "n_samples": {"train": 1158, "val": 165, "test": 310},
        "license": "CC BY 4.0",
    },
    "adrenalmnist3d": {
        "python_class": "AdrenalMNIST3D",
        "description": "The AdrenalMNIST3D is a new 3D shape classification dataset, consisting of shape masks from 1,584 left and right adrenal glands (i.e., 792 patients). Collected from Zhongshan Hospital Affiliated to Fudan University, each 3D shape of adrenal gland is annotated by an expert endocrinologist using abdominal computed tomography (CT), together with a binary classification label of normal adrenal gland or adrenal mass. Considering patient privacy, we do not provide the source CT scans, but the real 3D shapes of adrenal glands and their classification labels. We calculate the center of adrenal and resize the center-cropped 64mm×64mm×64mm volume into 28×28×28. The dataset is randomly split into training/validation/test set of 1,188/98/298 on a patient level.",
        "url": "https://zenodo.org/records/10519652/files/adrenalmnist3d.npz?download=1",
        "MD5": "bbd3c5a5576322bc4cdfea780653b1ce",
        "url_64": "https://zenodo.org/records/10519652/files/adrenalmnist3d_64.npz?download=1",
        "MD5_64": "17721accfe9fb005146a47d33bc54b2f",
        "task": "binary-class",
        "label": {"0": "normal", "1": "hyperplasia"},
        "n_channels": 1,
        "n_samples": {"train": 1188, "val": 98, "test": 298},
        "license": "CC BY 4.0",
    },
    "fracturemnist3d": {
        "python_class": "FractureMNIST3D",
        "description": "The FractureMNIST3D is based on the RibFrac Dataset, containing around 5,000 rib fractures from 660 computed tomography 153 (CT) scans. The dataset organizes detected rib fractures into 4 clinical categories (i.e., buckle, nondisplaced, displaced, and segmental rib fractures). As we use low-resolution images, we disregard segmental rib fractures and classify 3 types of rib fractures (i.e., buckle, nondisplaced, and displaced). For each annotated fracture area, we calculate its center and resize the center-cropped 64mm×64mm×64mm image into 28×28×28. The official split of training, validation and test set is used.",
        "url": "https://zenodo.org/records/10519652/files/fracturemnist3d.npz?download=1",
        "MD5": "6aa7b0143a6b42da40027a9dda61302f",
        "url_64": "https://zenodo.org/records/10519652/files/fracturemnist3d_64.npz?download=1",
        "MD5_64": "f01d7e6316aedf4210da0da5b7437b42",
        "task": "multi-class",
        "label": {
            "0": "buckle rib fracture",
            "1": "nondisplaced rib fracture",
            "2": "displaced rib fracture",
        },
        "n_channels": 1,
        "n_samples": {"train": 1027, "val": 103, "test": 240},
        "license": "CC BY 4.0",
    },
    "vesselmnist3d": {
        "python_class": "VesselMNIST3D",
        "description": "The VesselMNIST3D is based on an open-access 3D intracranial aneurysm dataset, IntrA, containing 103 3D models (meshes) of entire brain vessels collected by reconstructing MRA images. 1,694 healthy vessel segments and 215 aneurysm segments are generated automatically from the complete models. We fix the non-watertight mesh with PyMeshFix and voxelize the watertight mesh with trimesh into 28×28×28 voxels. We split the source dataset with a ratio of 7:1:2 into training, validation and test set.",
        "url": "https://zenodo.org/records/10519652/files/vesselmnist3d.npz?download=1",
        "MD5": "b41fd4f7e7e2feedddb201585ecafa1b",
        "url_64": "https://zenodo.org/records/10519652/files/vesselmnist3d_64.npz?download=1",
        "MD5_64": "6bb274a8846e1097066dcd64e2c4520f",
        "task": "binary-class",
        "label": {"0": "vessel", "1": "aneurysm"},
        "n_channels": 1,
        "n_samples": {"train": 1335, "val": 191, "test": 382},
        "license": "CC BY 4.0",
    },
    "synapsemnist3d": {
        "python_class": "SynapseMNIST3D",
        "description": "The SynapseMNIST3D is a new 3D volume dataset to classify whether a synapse is excitatory or inhibitory. It uses a 3D image volume of an adult rat acquired by a multi-beam scanning electron microscope. The original data is of the size 100×100×100um^3 and the resolution 8×8×30nm^3, where a (30um)^3 sub-volume was used in the MitoEM dataset with dense 3D mitochondria instance segmentation labels. Three neuroscience experts segment a pyramidal neuron within the whole volume and proofread all the synapses on this neuron with excitatory/inhibitory labels. For each labeled synaptic location, we crop a 3D volume of 1024×1024×1024nm^3 and resize it into 28×28×28 voxels. Finally, the dataset is randomly split with a ratio of 7:1:2 into training, validation and test set.",
        "url": "https://zenodo.org/records/10519652/files/synapsemnist3d.npz?download=1",
        "MD5": "1235b78a3cd6280881dd7850a78eadb6",
        "url_64": "https://zenodo.org/records/10519652/files/synapsemnist3d_64.npz?download=1",
        "MD5_64": "43bd14ebf3af9d3dd072446fedc14d5e",
        "task": "binary-class",
        "label": {"0": "inhibitory synapse", "1": "excitatory synapse"},
        "n_channels": 1,
        "n_samples": {"train": 1230, "val": 177, "test": 352},
        "license": "CC BY 4.0",
    },
    "camelyon17": {
        "label": {"0": "negative tumor", "1": "positive tumor"}
    },
    "drishti": {
        "label": {"0": "normal retina ", "1": "glaucomatous retina"}
    },
    "ham10000": {
        "label": {"0": "benign keratosis-like lesions", "1": "actinic keratoses or melanoma"}
    },
    "chestxray": {
        "label": {
            "0": "chest xray with no finding", "1": " chest xray with pneumonia"
        }
    },
    "gf3300":{
        "label" : {
            "0" : "benign retina", "1": "glaucomatous retina"
        }
    },
    "chexpert":{
        "label" : {
            "0" : "diseased xray", "1": "normal xray"
        }
    },
    "papila":{
        "label" : {
            "0" : "benign fundus", "1": "glaucoma fundus"
        }
    },
    "harvardfairvlmed10k":{
        "label" : {
            "0" : "a benign scanning laser ophthalmoscope fundus image", "1": " a glaucoma scanning laser ophthalmoscope fundus image"
        }
    }
}

class MedMNIST(Dataset):
    flag = ...

    def __init__(
        self,
        data_args,
        split,
        transform=None,
        target_transform=None,
        download=True,
        as_rgb=True,
        size=224,
        mmap_mode=None,
    ):
        """
        Args:

            split (string): 'train', 'val' or 'test', required
            transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version. Default: None.
            target_transform (callable, optional): A function/transform that takes in the target and transforms it. Default: None.
            download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory. If dataset is already downloaded, it is not downloaded again. Default: False.
            as_rgb (bool, optional): If true, convert grayscale images to 3-channel images. Default: False.
            size (int, optional): The size of the returned images. If None, use MNIST-like 28. Default: None.
            mmap_mode (str, optional): If not None, read image arrays from the disk directly. This is useful to set `mmap_mode='r'` to save memory usage when the dataset is large (e.g., PathMNIST-224). Default: None.
            root (string, optional): Root directory of dataset. Default: `~/.medmnist`.

        """
        # Here, `size_flag` is blank for 28 images, and `_size` for larger images, e.g., "_64".
        if (size is None) or (size == 28):
            self.size = 28
            self.size_flag = ""
        else:
            assert size in self.available_sizes
            self.size = size
            self.size_flag = f"_{size}"

        self.info = INFO[self.flag]
        self.name = self.flag

        if data_args.image_path is not None and os.path.exists(data_args.image_path):
            self.root = data_args.image_path
        else:
            raise RuntimeError(
                "Failed to setup the default `root` directory. "
                + "Please specify and create the `root` directory manually."
            )

        if download:
            self.download()

        if not os.path.exists(
            os.path.join(self.root, f"{self.flag}{self.size_flag}.npz")
        ):
            raise RuntimeError(
                "Dataset not found. " + " You can set `download=True` to download it"
            )

        npz_file = np.load(
            os.path.join(self.root, f"{self.flag}{self.size_flag}.npz"),
            mmap_mode=mmap_mode,
        )

        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.as_rgb = as_rgb

        if self.split in ["train", "val", "test"]:
            self.imgs = npz_file[f"{self.split}_images"]
            self.labels = npz_file[f"{self.split}_labels"]
        else:
            raise ValueError

    def __len__(self):
        assert self.info["n_samples"][self.split] == self.imgs.shape[0]
        return self.imgs.shape[0]

    def __repr__(self):
        """Adapted from torchvision."""
        _repr_indent = 4
        head = f"Dataset {self.__class__.__name__} of size {self.size} ({self.flag}{self.size_flag})"
        body = [f"Number of datapoints: {self.__len__()}"]
        body.append(f"Root location: {self.root}")
        body.append(f"Split: {self.split}")
        body.append(f"Task: {self.info['task']}")
        body.append(f"Number of channels: {self.info['n_channels']}")
        body.append(f"Meaning of labels: {self.info['label']}")
        body.append(f"Number of samples: {self.info['n_samples']}")
        body.append(f"Description: {self.info['description']}")
        body.append(f"License: {self.info['license']}")

        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)

    def download(self):
        try:
            from torchvision.datasets.utils import download_url

            download_url(
                url=self.info[f"url{self.size_flag}"],
                root=self.root,
                filename=f"{self.flag}{self.size_flag}.npz",
                md5=self.info[f"MD5{self.size_flag}"],
            )
        except:
            raise RuntimeError(
                f"""
                Automatic download failed! Please download {self.flag}{self.size_flag}.npz manually.
                1. [Optional] Check your network connection: 
                    Go to {HOMEPAGE} and find the Zenodo repository
                2. Download the npz file from the Zenodo repository or its Zenodo data link: 
                    {self.info[f"url{self.size_flag}"]}
                3. [Optional] Verify the MD5: 
                    {self.info[f"MD5{self.size_flag}"]}
                4. Put the npz file under your MedMNIST root folder: 
                    {self.root}
                """
            )


class MedMNIST2D(MedMNIST):
    available_sizes = [28, 64, 128, 224]

    def __getitem__(self, index):
        """
        return: (without transform/target_transofrm)
            img: PIL.Image
            target: np.array of `L` (L=1 for single-label)
        """
        img, label = self.imgs[index], self.labels[index].astype(int)
        img = Image.fromarray(img)

        if self.as_rgb:
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return {
            "pixel_values": img,
            "label": label,
            "image_path": ""
        }

    def save(self, folder, postfix="png", write_csv=True):
        from medmnist.utils import save2d

        save2d(
            imgs=self.imgs,
            labels=self.labels,
            img_folder=os.path.join(folder, f"{self.flag}{self.size_flag}"),
            split=self.split,
            postfix=postfix,
            csv_path=os.path.join(folder, f"{self.flag}{self.size_flag}.csv")
            if write_csv
            else None,
        )

    def montage(self, length=20, replace=False, save_folder=None):
        from medmnist.utils import montage2d

        n_sel = length * length
        sel = np.random.choice(self.__len__(), size=n_sel, replace=replace)

        montage_img = montage2d(
            imgs=self.imgs, n_channels=self.info["n_channels"], sel=sel
        )

        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            montage_img.save(
                os.path.join(
                    save_folder, f"{self.flag}{self.size_flag}_{self.split}_montage.jpg"
                )
            )

        return montage_img


class MedMNIST3D(MedMNIST):
    available_sizes = [28, 64]

    def __getitem__(self, index):
        """
        return: (without transform/target_transofrm)
            img: an array of 1x28x28x28 or 3x28x28x28 (if `as_RGB=True`), in [0,1]
            target: np.array of `L` (L=1 for single-label)
        """
        img, label = self.imgs[index], self.labels[index].astype(int)

        img = np.stack([img / 255.0] * (3 if self.as_rgb else 1), axis=0)

        if self.transform is not None:
            img = self.transform(img)

        if label.target_transform is not None:
            target = self.target_transform(label)

        return {
            "pixel_values": img,
            "label": label,
            "image_path": ""
        }

    def save(self, folder, postfix="gif", write_csv=True):
        from medmnist.utils import save3d

        assert postfix == "gif"

        save3d(
            imgs=self.imgs,
            labels=self.labels,
            img_folder=os.path.join(folder, f"{self.flag}{self.size_flag}"),
            split=self.split,
            postfix=postfix,
            csv_path=os.path.join(folder, f"{self.flag}{self.size_flag}.csv")
            if write_csv
            else None,
        )

    def montage(self, length=20, replace=False, save_folder=None):
        assert self.info["n_channels"] == 1

        from medmnist.utils import montage3d, save_frames_as_gif

        n_sel = length * length
        sel = np.random.choice(self.__len__(), size=n_sel, replace=replace)

        montage_frames = montage3d(
            imgs=self.imgs, n_channels=self.info["n_channels"], sel=sel
        )

        if save_folder is not None:
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)

            save_frames_as_gif(
                montage_frames,
                os.path.join(
                    save_folder, f"{self.flag}{self.size_flag}_{self.split}_montage.gif"
                ),
            )

        return montage_frames


class PathMNIST(MedMNIST2D):
    flag = "pathmnist"


class OCTMNIST(MedMNIST2D):
    flag = "octmnist"


class PneumoniaMNIST(MedMNIST2D):
    flag = "pneumoniamnist"


class ChestMNIST(MedMNIST2D):
    flag = "chestmnist"


class DermaMNIST(MedMNIST2D):
    flag = "dermamnist"


class RetinaMNIST(MedMNIST2D):
    flag = "retinamnist"


class BreastMNIST(MedMNIST2D):
    flag = "breastmnist"


class BloodMNIST(MedMNIST2D):
    flag = "bloodmnist"


class TissueMNIST(MedMNIST2D):
    flag = "tissuemnist"


class OrganAMNIST(MedMNIST2D):
    flag = "organamnist"


class OrganCMNIST(MedMNIST2D):
    flag = "organcmnist"


class OrganSMNIST(MedMNIST2D):
    flag = "organsmnist"


class OrganMNIST3D(MedMNIST3D):
    flag = "organmnist3d"


class NoduleMNIST3D(MedMNIST3D):
    flag = "nodulemnist3d"


class AdrenalMNIST3D(MedMNIST3D):
    flag = "adrenalmnist3d"


class FractureMNIST3D(MedMNIST3D):
    flag = "fracturemnist3d"


class VesselMNIST3D(MedMNIST3D):
    flag = "vesselmnist3d"


class SynapseMNIST3D(MedMNIST3D):
    flag = "synapsemnist3d"


# Camelyon17 Dataset
class Camelyon17(Dataset):
    def __init__(self, data_args, metadata_path_train="./data/camelyon17_v1.0/sample3680_train_metadata.csv", metadata_path_test="./data/camelyon17_v1.0/sample920_test_metadata.csv", split="train", transform=None):
        self.image_root = data_args.image_path
        self.transform = transform
        self.split = split
        self.name = "Camelyon17"
        
        # Load the appropriate dataset
        metadata_path = metadata_path_train if self.split == "train" else metadata_path_test
        self.data = pd.read_csv(metadata_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_root, f"patient_{row['patient']:03d}_node_{row['node']}", f"patch_patient_{row['patient']:03d}_node_{row['node']}_x_{row['x_coord']}_y_{row['y_coord']}.png")
        img = Image.open(image_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        label = torch.tensor(row['tumor'], dtype=torch.long)
        
        return {
            "pixel_values": img,
            "label": label,
            "image_path": image_path
        }


class HAM10000Dataset(torch.utils.data.Dataset):
    def __init__(self, data_args, transform, split="train"):
        self.CLASSES = 2
        self.class_dict = {'benign keratosis-like lesions': 0, 'actinic keratoses or melanoma': 1}
        self.transform = transform
        self.meta_path = os.path.join(data_args.image_path, "HAM10000", "split", "{}.csv".format(split))
        self.df = pd.read_csv(self.meta_path)
        self.Y = self.df["dx"].values.copy()
        self.Y[self.Y == "akiec"] = 1
        self.Y[self.Y == "mel"] = 1
        self.Y[self.Y != 1] = 0
        self.path_to_images = os.path.join(data_args.image_path, "HAM10000", "HAM10000_images")
        self.name = "HAM10000"

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.path_to_images, f"{self.df.iloc[idx]['image_id']}.jpg"))
        label = torch.tensor(self.Y[idx])
        if self.transform is not None:
            image = self.transform(image)

        return {
            "pixel_values": image,
            "label": label
        }

    
class GF3300Dataset(torch.utils.data.Dataset):
    def __init__(self, data_args, transform, split="train"):
        self.CLASSES = 2
        self.class_dict = {'benign retina': 0, 'glaucoma retina': 1}
        self.transform = transform
        self.meta_path = os.path.join(data_args.image_path, "GF3300", "split", "{}.csv".format(split))
        self.df = pd.read_csv(self.meta_path)
        self.Y = np.where(
            self.df["glaucoma"] == "yes",
            1,
            0,
        )
        self.class_nums = 2
        self.path_to_images = os.path.join(data_args.image_path, "GF3300", "data")
        self.name = "GF3300"

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        # if self.path_to_labels is not None:
        #     image = Image.fromarray(self.tol_images[idx])
        # else:
        image = np.load(os.path.join(self.path_to_images, item["path"]))["rnflt"]
        image = (image - (-2)) / (350 - (-2)) * 255
        image = Image.fromarray(image.astype(np.uint8)).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(self.Y[idx])

        return {
            "pixel_values": image,
            "label": label
        }



class CXPDataset(torch.utils.data.Dataset):
    # Uses the original CheXpert dataset split. https://github.com/FairMedFM/FairMedFM/blob/main/pre-processing/classification/CXP.ipynb

    def __init__(
        self,
        data_args,
        transform=None,
        split: str = "train",
    ):
        split = "valid" if split == "test" else "train"
        self.CLASSES = 2
        self.class_dict = {f"no finding": 0, "has findings": 1}
        self.transform = transform

        # --- metadata ---
        self.meta_path = os.path.join(
            data_args.image_path, "CheXpert-v1.0-small", f"{split}.csv"
        )
        self.df = pd.read_csv(self.meta_path)

        # Treat uncertain (-1) or NaN as negative (0)
        labels = (
            self.df["No Finding"]
            .replace(-1, 0)
            .fillna(0)
            .astype(int)
            .to_numpy()
        )
        self.Y = np.where(labels == 1, 1, 0).astype(np.int64)

        self.class_nums = 2
        self.path_to_images = os.path.join(data_args.image_path)
        self.name = "CXP"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]

        img_path = os.path.join(self.path_to_images, item["Path"])
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(self.Y[idx])

        return {
            "pixel_values": image,
            "label": label,
        }
    


class PAPILADataset(torch.utils.data.Dataset):

    def __init__(self, data_args, transform, split: str = "train"):
        # ── labels ───────────────────────────────────────────────────────
        self.CLASSES = 2
        # normal → 0, glaucoma → 1  (edit if you have different naming)
        self.class_dict = {"normal": 0, "glaucoma": 1}

        # ── bookkeeping ─────────────────────────────────────────────────
        self.transform = transform
        self.name = "PAPILA"

        # CSV with file list + metadata
        self.meta_path = os.path.join(
            data_args.image_path, "PAPILA", "split", f"new_{split}.csv"
        )
        self.df = pd.read_csv(self.meta_path)

        self.Y = self.df["Diagnosis"].values.astype(np.int64)

        self.path_to_images = os.path.join(
            data_args.image_path, "PAPILA", "data", "FundusImages"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path_in_csv = str(self.df.iloc[idx]["Path"])
        img_path = os.path.join(self.path_to_images, path_in_csv)
        image = Image.open(img_path).convert("RGB")

        label = torch.tensor(self.Y[idx], dtype=torch.long)

        if self.transform is not None:
            image = self.transform(image)

        return {
            "pixel_values": image,
            "label": label
        }
    
class FairVLMed10kDataset(torch.utils.data.Dataset):

    def __init__(self, data_args, transform, split: str = "train"):
        self.CLASSES = 2
        self.class_dict = {"benign": 0, "glaucoma": 1}
        self.transform = transform
        self.name = "FairVLMed10k"

        self.meta_path = os.path.join(
            data_args.image_path, "FairVLMed10k", "split", f"{split}.csv"
        )
        self.df = pd.read_csv(self.meta_path)

        self.Y = np.where(self.df["glaucoma"].str.lower() == "yes", 1, 0).astype(
            np.int64
        )

        self.path_to_images = os.path.join(
            data_args.image_path, "FairVLMed10k"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_rel_path = str(self.df.iloc[idx]["path"])
        img_path = os.path.join(self.path_to_images, img_rel_path)
        image = Image.open(img_path).convert("RGB")

        label = torch.tensor(self.Y[idx], dtype=torch.long)

        if self.transform is not None:
            image = self.transform(image)

        return {
            "pixel_values": image,
            "label": label
        }
    

class DrishtiDataset(torch.utils.data.Dataset):
    def __init__(self, data_args, transform, split="train"):
        self.name = 'DrishtiDataset'
        self.CLASSES = 2
        self.class_dict = {'Normal': 0, 'Glaucomatous': 1}
        self.transform = transforms.Compose([
            transforms.PILToTensor(),  # Convert PIL Image to Tensor
            transforms.Resize((224, 224))  # Resize the Tensor to 128x128
        ])
        if split == "train":
            self.file = os.path.join(data_args.image_path, 'Drishti-GS1_files', 'Drishti-GS1_files', 'Training', 'Images')
        elif split == "test":
            self.file = os.path.join(data_args.image_path, 'Drishti-GS1_files', 'Drishti-GS1_files', 'Test', 'Images')
        else:
            raise RuntimeError("Split must be one of train and test.")

        self.labels = self.get_label(os.path.join(data_args.image_path, 'Drishti-GS1_files', 'Drishti-GS1_files', 'Drishti-GS1_diagnosis.csv'))
        self.image_names = []
        self.label_tensors = []

        for img_name in os.listdir(self.file):
            self.image_names.append(os.path.join(self.file,img_name))
            self.label_tensors.append(torch.tensor(self.class_dict[self.labels[img_name[:-4]]], dtype=torch.long))

    def get_label(self, file):
        retVal = {}
        lines = list(csv.reader(open(file)))
        for line in lines[1:]:
            retVal[line[0][:-1]] = line[-2]
        return retVal

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image = Image.open(self.image_names[index]).convert('RGB')
        label = self.label_tensors[index]

        if self.transform is not None:
            image = self.transform(image)

        return {
            "pixel_values": image,
            "label": label
        }


class ChestXrayDataset(Dataset):
    def __init__(self, data_args, transform=None, split="test"):
        """
        Args:
            root_dir (str): Path to the dataset root directory (e.g., 'chest_xray').
            transform (callable, optional): Transform to be applied to the images.
            split (str): The dataset split to load ('train' or 'test').
        """
        self.root_dir = os.path.join(data_args.image_path, split)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        # Class mapping
        self.class_dict = {'NORMAL': 0, 'PNEUMONIA': 1}
        
        # Collect image paths and labels
        self.image_paths = []
        self.labels = []
        self.name = "ChestXray"
        
        for class_name, label in self.class_dict.items():
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.labels[index], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return {
            "pixel_values": image,
            "label": label
        }

# backward-compatible aliases
OrganMNISTAxial = OrganAMNIST
OrganMNISTCoronal = OrganCMNIST
OrganMNISTSagittal = OrganSMNIST


if __name__ == "__main__":
    DEFAULT_ROOT = '/fast/rjin02/MedVLMBench/data'
    transform = transforms.Compose([
        transforms.PILToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])  # Align with FairMedFM
    ])



    dataset = PneumoniaMNIST(
        split="val", # 'train', 'val' or 'test'
        transform=None,
        target_transform=None,
        download=True,
        as_rgb=True,
        root=DEFAULT_ROOT,
        size=224,
        mmap_mode=None,
    )