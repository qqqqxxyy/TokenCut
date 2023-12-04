"""
Datasets file. Code adapted from LOST: https://github.com/valeoai/LOST
"""
from email.mime import image
import os
import torch
import json
import torchvision
import numpy as np
import skimage.io
import h5py
from PIL import Image
from tqdm import tqdm
from skimage.transform import resize
from torchvision import transforms as pth_transforms
from datasets_flexible import VOCDetection
import math
import ipdb
from utils.IO import load_path_file


# PATH_FILE1 = os.path.abspath(os.path.split(__file__)[0]+'../../../../weight/paths.json')
# def load_path_file():
#     with open(PATH_FILE1,'r') as f:
#         path_file = json.load(f)
#     return path_file
PATH_FILE = load_path_file()
DATA_PATH = PATH_FILE['dataset_path']
# Image transformation applied to all images
transform = pth_transforms.Compose(
    [
        # pth_transforms.Resize((448,448)), 
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

transform_rot = pth_transforms.Compose(
    [
        # pth_transforms.Resize((448,448)), 
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

class ImageDataset:
    def __init__(self, image_path, resize=None):
        
        self.image_path = image_path
        self.name = image_path.split("/")[-1]

        # Read the image
        with open(image_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")

        # Build a dataloader
        if resize is not None:
            transform_resize = pth_transforms.Compose(
                [ 
                    pth_transforms.ToTensor(),
                    pth_transforms.Resize(resize),
                    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
            img = transform_resize(img)
            self.img_size = list(img.shape[-1:-3:-1])
        else:
            img = transform(img)
            self.img_size = list(img.shape[-1:-3:-1])
        self.dataloader = [[img, image_path]]

    def get_image_name(self, *args, **kwargs):
        return self.image_path.split("/")[-1].split(".")[0]

    def load_image(self, *args, **kwargs):
        return Image.open(self.image_path).convert("RGB").resize(self.img_size)

class Dataset:
    def __init__(self, dataset_name, dataset_set, remove_hards):
        """
        Build the dataloader
        """

        self.dataset_name = dataset_name
        self.set = dataset_set

        if dataset_name == "VOC07":
            self.root_path = os.path.join(DATA_PATH,'VOC2007/data/JPEGImages')
            self.year = "2007"
        elif dataset_name == "VOC12":
            self.root_path = os.path.join(DATA_PATH,'VOC2012/data/JPEGImages')
            # self.root_path = "/home/qxy/Desktop/datasets/VOC2012/data/Sample"
            self.year = "2012"
        elif dataset_name == "COCO20k":
            self.year = "2014"
            self.root_path = f"datasets/COCO/images/{dataset_set}{self.year}"
            self.sel20k = 'datasets/coco_20k_filenames.txt'
            # JSON file constructed based on COCO train2014 gt 
            self.all_annfile = "datasets/COCO/annotations/instances_train2014.json"
            self.annfile = "datasets/instances_train2014_sel20k.json"
            self.sel_20k = get_sel_20k(self.sel20k)
            if not os.path.exists(self.annfile):
                select_coco_20k(self.sel20k, self.all_annfile)
            self.train2014 = get_train2014(self.annfile)
        else:
            raise ValueError("Unknown dataset.")

        if not os.path.exists(self.root_path):
            raise ValueError("Please follow the README to setup the datasets.")

        self.name = f"{self.dataset_name}_{self.set}"

        # Build the dataloader
        if "VOC" in dataset_name:
            self.dataloader = VOCDetection(
                self.root_path,
                year=self.year,
                image_set=self.set,
                transform=transform,
                download=False,
            )
        elif "COCO20k" == dataset_name:
            self.dataloader = torchvision.datasets.CocoDetection(
                self.root_path, annFile=self.annfile, transform=transform
            )
        else:
            raise ValueError("Unknown dataset.")

        # Set hards images that are not included
        self.remove_hards = remove_hards
        self.hards = []
        if remove_hards:
            self.name += f"-nohards"
            self.hards = self.get_hards()
            print(f"Nb images discarded {len(self.hards)}")


    def load_image(self, im_name):
        """
        Load the image corresponding to the im_name
        """
        if "VOC" in self.dataset_name:
            # image = skimage.io.imread(f"/home/qxy/Desktop/datasets/VOC{self.year}/VOCdevkit/VOC{self.year}/JPEGImages/{im_name}")
            image = skimage.io.imread(f"/home/qxy/Desktop/datasets/VOC{self.year}/VOCdevkit/VOC{self.year}/Sample/{im_name}")

        elif "COCO" in self.dataset_name:
            #im_path = self.path_20k[self.sel_20k.index(im_name)]
            #im_path = self.train2014['images'][self.sel_20k.index(im_name)]['file_name']
            #image = skimage.io.imread(f"./datasets/COCO/images/train2014/{im_path}")
            image = skimage.io.imread(f"./datasets/COCO/images/train2014/{im_name}")
        else:
            raise ValueError("Unkown dataset.")
        return image

    def get_image_name(self, inp):
        """
        Return the image name
        """
        if "VOC" in self.dataset_name:
            im_name = inp["annotation"]["filename"]
        elif "COCO" in self.dataset_name:
            im_name = str(inp[0]["image_id"])
            im_name = self.train2014['images'][self.sel_20k.index(im_name)]['file_name']

        return im_name

    def extract_gt(self, targets, im_name):
        if "VOC" in self.dataset_name:
            return extract_gt_VOC(targets, remove_hards=self.remove_hards)
        elif "COCO" in self.dataset_name:
            return extract_gt_COCO(targets, remove_iscrowd=True)
        else:
            raise ValueError("Unknown dataset")

    def extract_classes(self):
        if "VOC" in self.dataset_name:
            cls_path = f"classes_{self.set}_{self.year}.txt"
        elif "COCO" in self.dataset_name:
            cls_path = f"classes_{self.dataset}_{self.set}_{self.year}.txt"

        # Load if exists
        if os.path.exists(cls_path):
            all_classes = []
            with open(cls_path, "r") as f:
                for line in f:
                    all_classes.append(line.strip())
        else:
            print("Extract all classes from the dataset")
            if "VOC" in self.dataset_name:
                all_classes = self.extract_classes_VOC()
            elif "COCO" in self.dataset_name:
                all_classes = self.extract_classes_COCO()

            with open(cls_path, "w") as f:
                for s in all_classes:
                    f.write(str(s) + "\n")

        return all_classes

    def extract_classes_VOC(self):
        all_classes = []
        for im_id, inp in enumerate(tqdm(self.dataloader)):
            objects = inp[1]["annotation"]["object"]

            for o in range(len(objects)):
                if objects[o]["name"] not in all_classes:
                    all_classes.append(objects[o]["name"])

        return all_classes

    def extract_classes_COCO(self):
        all_classes = []
        for im_id, inp in enumerate(tqdm(self.dataloader)):
            objects = inp[1]

            for o in range(len(objects)):
                if objects[o]["category_id"] not in all_classes:
                    all_classes.append(objects[o]["category_id"])

        return all_classes

    def get_hards(self):
        hard_path = "datasets/hard_%s_%s_%s.txt" % (self.dataset_name, self.set, self.year)
        if os.path.exists(hard_path):
            hards = []
            with open(hard_path, "r") as f:
                for line in f:
                    hards.append(int(line.strip()))
        else:
            print("Discover hard images that should be discarded")

            if "VOC" in self.dataset_name:
                # set the hards
                hards = discard_hard_voc(self.dataloader)

            with open(hard_path, "w") as f:
                for s in hards:
                    f.write(str(s) + "\n")

        return hards


class dataset_file_load(object):
    #def __init__(self):
        #self.file_path = file_path
        #self.root_path = root_path

    def read_path_list(self,file_path,root_path,mask=False):
        '''
        由于mask路径要把.jpg改成.png,因此要做额外的处理
        '''
        relative_path = self._read_half_path(file_path,mask)
        absolute_path = self._concat_path_list(relative_path,root_path)
        
        return absolute_path

    def read_key_list(self,file_path,key):
        '''读取file_path中文件属性中关于key属性的list
        '''
        with open(file_path,'r') as f:
            property_list = json.load(f)
        
        key_list = []
        for dic in property_list:
            key_list.append(dic[key])    
        return key_list            

    def _read_half_path(self,file_path,mask):
        if file_path == None:
            file_path = self.file_path

        with open(file_path,'r') as f:
            property_list = json.load(f)
        path_list = []
        for img_property in property_list:
            if mask == False:
                path_list.append(img_property['path'])
            else:
                path_list.append(img_property['path'].replace('jpg','png'))
        return path_list

    def _concat_path_list(self,path_list,root_path):
        absolute_path=[]
        for paths in path_list:
            paths = os.path.join(root_path,paths)
            absolute_path.append(paths)
        return absolute_path

class OriDataset(Dataset):
    '''the base function of dataset, including: 
    image loading, class loading, tsfm of image
    input:
    images_root: string variable, the path of dataset
    image_prop: string variable, the path of property: json file
    tsfm: dict funtion, {'resize':tuple, 'normalize': /; 'crop':/}
    output:
    idx: the index is called for
    img: tensor (w,h,3)
    clas: tensor (0)
    '''
    def __init__(self,images_root,image_prop,tsfm=transform,mask=False,hdf5=False):
        self.load_tool = dataset_file_load()
        #read image path list
        
        self.hdf5 = hdf5
        if self.hdf5 == True:
            #images_root: 索取要读取的目录文件
            #images_path_list: 文件名
            self.image_path_list = self.load_tool._read_half_path(image_prop,mask)
            self.images_root = images_root+'.hdf'
            # self.images_root = '/mnt/usb/Dataset relate/ILSVRC_hdf5/data.hdf'
        else:
            self.image_path_list = self.load_tool.read_path_list(image_prop,images_root,mask)
        #read class
        #-1 because in property.json, the class is started with 1
        self.clas_list = torch.tensor(self.load_tool.read_key_list(image_prop,'class'))-1
        self.transform  = tsfm


    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self,idx):
        if self.hdf5 == True:
            img = self._read_hdf5(idx)
        else:
            img = Image.open(self.image_path_list[idx])

        if self.transform != None:
            img = self.transform(img.convert('RGB'))
        clas  = self.clas_list[idx]

        return idx, img, clas

    def _read_hdf5(self,idx):
        image_name = self.image_path_list[idx]
        with h5py.File(self.images_root,'r') as hdf:
            img =  np.array(hdf.get(image_name))
            img = Image.fromarray(img)
        return img


class ObjLocDataset(OriDataset):
    'extra output bounding box based on OriDataset'  
    '''
    tsfm: transform for the image, if None, then transform will be transform (defined above)
    '''  
    def __init__(self,dataset,set,tsfm=transform,tsfm_bbox=None,hdf5=False):
        images_root,image_prop = self._generate_path(dataset,set)
        super(ObjLocDataset, self).__init__(images_root,image_prop,tsfm,hdf5=hdf5)
        self.bbox_list = self.load_tool.read_key_list(image_prop,'bbox')
        self.size_list = self.load_tool.read_key_list(image_prop,'size')
        self.tsfm_bbox = tsfm_bbox

    def _generate_path(self,dataset,set):
        if dataset == 'CUB':
            images_root = os.path.join(DATA_PATH,'CUB/data')
            images_proerty = os.path.join(DATA_PATH,'CUB/data_annotation/{}.json'.format(set))
        return images_root,images_proerty

    def load_image(self, im_name):
        image = skimage.io.imread(self.image_path_list[im_name])
        return image

    def __getitem__(self,idx):
        '''
        output:
        image: tensor (3,w,h)
        clas: tensor (0)
        bbox: float list [x0,y0,x1,y1]
        '''
        idx,img,clas = super(ObjLocDataset, self).__getitem__(idx)
        if self.tsfm_bbox != None:
            bbox,_ = self.tsfm_bbox([self.bbox_list[idx],self.size_list[idx][0]])
        else:
            bbox = self.bbox_list[idx]
        return img,bbox


def discard_hard_voc(dataloader):
    hards = []
    for im_id, inp in enumerate(tqdm(dataloader)):
        objects = inp[1]["annotation"]["object"]
        nb_obj = len(objects)

        hard = np.zeros(nb_obj)
        for i, o in enumerate(range(nb_obj)):
            hard[i] = (
                1
                if (objects[o]["truncated"] == "1" or objects[o]["difficult"] == "1")
                else 0
            )

        # all images with only truncated or difficult objects
        if np.sum(hard) == nb_obj:
            hards.append(im_id)
    return hards


def extract_gt_COCO(targets, remove_iscrowd=True):
    objects = targets
    nb_obj = len(objects)

    gt_bbxs = []
    gt_clss = []
    for o in range(nb_obj):
        # Remove iscrowd boxes
        if remove_iscrowd and objects[o]["iscrowd"] == 1:
            continue
        gt_cls = objects[o]["category_id"]
        gt_clss.append(gt_cls)
        bbx = objects[o]["bbox"]
        x1y1x2y2 = [bbx[0], bbx[1], bbx[0] + bbx[2], bbx[1] + bbx[3]]
        x1y1x2y2 = [int(round(x)) for x in x1y1x2y2]
        gt_bbxs.append(x1y1x2y2)

    return np.asarray(gt_bbxs), gt_clss


def extract_gt_VOC(targets, remove_hards=False,resize=False):
    objects = targets["annotation"]["object"]
    nb_obj = len(objects)

    gt_bbxs = []
    gt_clss = []
    for o in range(nb_obj):
        if remove_hards and (
            objects[o]["truncated"] == "1" or objects[o]["difficult"] == "1"
        ):
            continue
        try:
            gt_cls = objects[o]["name"]
            gt_clss.append(gt_cls)
            obj = objects[o]["bndbox"]
        except:
            gt_cls = objects["name"]
            gt_clss.append(gt_cls)
            obj = objects["bndbox"]

        x1y1x2y2 = [
            int(obj["xmin"]),
            int(obj["ymin"]),
            int(obj["xmax"]),
            int(obj["ymax"]),
        ]
        # Original annotations are integers in the range [1, W or H]
        # Assuming they mean 1-based pixel indices (inclusive),
        # a box with annotation (xmin=1, xmax=W) covers the whole image.
        # In coordinate space this is represented by (xmin=0, xmax=W)
        x1y1x2y2[0] -= 1
        x1y1x2y2[1] -= 1
        gt_bbxs.append(x1y1x2y2)
        if resize:
            w = resize[0]/int(targets["annotation"]["size"]["width"])
            h = resize[1]/int(targets["annotation"]["size"]["height"])
            x1y1x2y2[0],x1y1x2y2[2]=x1y1x2y2[0]*w,x1y1x2y2[2]*w
            x1y1x2y2[1],x1y1x2y2[3]=x1y1x2y2[1]*h,x1y1x2y2[3]*h
    return np.asarray(gt_bbxs), gt_clss


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # https://github.com/ultralytics/yolov5/blob/develop/utils/general.py
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T
    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(
            b1_x1, b2_x1
        )  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
                + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
            ) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif (
                CIoU
            ):  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(
                    torch.atan(w2 / h2) - torch.atan(w1 / h1), 2
                )
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU

def get_sel_20k(sel_file):  
    # load selected images
    with open(sel_file, "r") as f:
        sel_20k = f.readlines()
        sel_20k = [s.replace("\n", "") for s in sel_20k]
    im20k = [str(int(s.split("_")[-1].split(".")[0])) for s in sel_20k]
    return im20k

def get_train2014(all_annotations_file):
    # load all annotations
    with open(all_annotations_file, "r") as f:
        train2014 = json.load(f)
    return train2014



def select_coco_20k(sel_file, all_annotations_file):
    print('Building COCO 20k dataset.')

    # load all annotations
    with open(all_annotations_file, "r") as f:
        train2014 = json.load(f)

    # load selected images
    with open(sel_file, "r") as f:
        sel_20k = f.readlines()
        sel_20k = [s.replace("\n", "") for s in sel_20k]
    im20k = [str(int(s.split("_")[-1].split(".")[0])) for s in sel_20k]

    new_anno = []
    new_images = []

    for i in tqdm(im20k):
        new_anno.extend(
            [a for a in train2014["annotations"] if a["image_id"] == int(i)]
        )
        new_images.extend([a for a in train2014["images"] if a["id"] == int(i)])
    
    train2014_20k = {}
    train2014_20k["images"] = new_images
    train2014_20k["annotations"] = new_anno
    train2014_20k["categories"] = train2014["categories"]

    with open("datasets/instances_train2014_sel20k.json", "w") as outfile:
        json.dump(train2014_20k, outfile)
    
    print(f'im20k :{im20k[0]}')
    print('Done.')
