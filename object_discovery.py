"""
Main functions for applying Normalized Cut.
Code adapted from LOST: https://github.com/valeoai/LOST
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy.linalg import eigh
from scipy import ndimage
import ipdb
from sklearn.decomposition import PCA,KernelPCA
from sklearn.preprocessing import normalize
def feat_extract(feats,dims,channel):
    '''Extract the NO.channel of the feats and reshape into its original dims. 
    '''
    feats = feats[0,1:,channel]
    feats = feats.unsqueeze(0)
    feats = F.normalize(feats, p=2)
    feats = feats.reshape(dims)
    return feats

def ncut(feats, dims, scales, init_image_size, tau = 0, eps=1e-5, no_binary_graph=False):
    """
    Implementation of NCut Method.
    Inputs
      feats: the pixel/patche features of an image
      dims: dimension of the map from which the features are used
      scales: from image to map scale
      init_image_size: size of the image
      tau: thresold for graph construction
      eps: graph edge weight
      im_name: image_name
      no_binary_graph: ablation study for using similarity score as graph edge weight
    """
    
    # feats = F.normalize(feats, p=2)
    # feats = feats.cpu().numpy()
    # print(feats[0][0])
    # print(feats.shape)
    # ipdb.set_trace()
    cls_token = feats[0,0:1,:]#.cpu().numpy() 
    feats = feats[0,1:,:]
    feats = F.normalize(feats, dim=1,p=2)
    

    
    # #PCA
    # feat = feats.cpu().numpy()
    # pca = PCA(n_components=10, random_state=42)
    # feat = pca.fit_transform(feat)
    # feat = normalize(feat,norm='l2',axis=1,copy=True,return_norm=False)
    
    # feat = torch.tensor(feat).cuda()
    
    # feats = torch.cat([feats1,feat[:,:2]],dim=1)

    A0 = (feats @ feats.transpose(1,0)) 

    # ipdb.set_trace()

    # A1 = (feat @ feat.transpose(1,0))
    # ipdb.set_trace()
    A = A0 #max(A0, # (A0-A0.min())/(A0.max()-A0.min())
    # ipdb.set_trace()
    A = A.cpu().numpy()

    # eigenvec = A
    
    
    if no_binary_graph:
        A[A<tau] = eps
    else:
        A = A > tau
        A = np.where(A.astype(float) == 0, eps, A)
    d_i = np.sum(A, axis=1)
    D = np.diag(d_i)
  
    # Print second and third smallest eigenvector 
    _, eigenvectors = eigh(D-A, D, subset_by_index=[1,2])
    eigenvec =  np.copy(eigenvectors[:, 0])

    # Using average point to compute bipartition 
    second_smallest_vec = eigenvectors[:, 0]
    
    avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
    bipartition = second_smallest_vec > avg
    
    seed = np.argmax(np.abs(second_smallest_vec))

    if bipartition[seed] != 1:
        eigenvec = eigenvec * -1 
        bipartition = np.logical_not(bipartition)

    bipartition = bipartition.reshape(dims).astype(float)
    # predict BBox
    pred, _, objects,cc = detect_box(bipartition, seed, dims, scales=scales, initial_im_size=init_image_size[1:]) ## We only extract the principal object BBox
    mask = np.zeros(dims)
    mask[cc[0],cc[1]] = 1
    # ipdb.set_trace()
    return np.asarray(pred), objects, mask, seed, None, eigenvec.reshape(dims)
    #(dims[0]*dims[1],dims[0]*dims[1])


def ncut2(feats, dims, scales, init_image_size, tau = 0, eps=1e-5, no_binary_graph=False):
    """
    Implementation of NCut Method.
    Inputs
      feats: the pixel/patche features of an image
      dims: dimension of the map from which the features are used
      scales: from image to map scale
      init_image_size: size of the image
      tau: thresold for graph construction
      eps: graph edge weight
      im_name: image_name
      no_binary_graph: ablation study for using similarity score as graph edge weight
    """
    # feats = F.normalize(feats, p=2)
    # feats = feats.cpu().numpy()
    # print(feats[0][0])
    # print(feats.shape)
    # ipdb.set_trace()
    cls_token = feats[0,0:1,:]#.cpu().numpy() 
    feats = feats[0,1:,:]
    # ipdb.set_trace()
    feats = F.normalize(feats, dim=1,p=2)
    # #PCA
    # feat = feats.cpu().numpy()
    # pca = PCA(n_components=10, random_state=42)
    # feat = pca.fit_transform(feat)
    # feat = normalize(feat,norm='l2',axis=1,copy=True,return_norm=False)
    # feat = torch.tensor(feat).cuda()
    # feats = torch.cat([feats1,feat[:,:2]],dim=1)

    A1 = (feats @ feats.transpose(1,0)) 
    # A0 = (A1 @ A1.transpose(1,0)) 
    A = A1 #max(A0, # (A0-A0.min())/(A0.max()-A0.min())
    # ipdb.set_trace()
    A = A.cpu().numpy()
    # eigenvec = A
    if no_binary_graph:
        A[A<tau] = eps
    else:
        A = A > tau
        A = np.where(A.astype(float) == 0, eps, A)
    d_i = np.sum(A, axis=1)
    D = np.diag(d_i)
    # Print second and third smallest eigenvector 
    eigenval, eigenvectors = eigh(D-A, D, subset_by_index=[1,2])
    W = torch.tensor(D-A)
    
    eigenvec =  np.copy(eigenvectors[:, 0])
    # Using average point to compute bipartition 
    second_smallest_vec = eigenvectors[:, 0]
    
    seed = np.argmax(np.abs(second_smallest_vec))
    if second_smallest_vec[seed]<0:
        second_smallest_vec = second_smallest_vec * -1
    
    avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
    bipartition = second_smallest_vec < avg
    if bipartition[seed] != 1:
        eigenvec = eigenvec * -1
        bipartition = np.logical_not(bipartition)
    #打印相似性矩阵图
    
    # bipartition_0 = torch.tensor(bipartition).bool()

 
    # feats = feats[ bipartition_0,:]
    # A0 = (feats @ feats.transpose(1,0)) 
    # A = A0 #max(A0, # (A0-A0.min())/(A0.max()-A0.min())
    # # ipdb.set_trace()
    # A = A.cpu().numpy()
    # # eigenvec = A
    # if no_binary_graph:
    #     A[A<tau] = eps
    # else:
    #     A = A > tau
    #     A = np.where(A.astype(float) == 0, eps, A)
    # d_i = np.sum(A, axis=1)
    # D = np.diag(d_i)
    # # Print second and third smallest eigenvector 
    # eigenval, eigenvectors = eigh(D-A, D, subset_by_index=[1,2])
    
    # eigenvec =  np.copy(eigenvectors[:, 0])
    # # Using average point to compute bipartition 
    # second_smallest_vec = eigenvectors[:, 0]
    
    # seed = np.argmax(np.abs(second_smallest_vec))
    # if second_smallest_vec[seed]<0:
    #     second_smallest_vec = second_smallest_vec * -1
    
    # avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
    # bipartition_1 = second_smallest_vec > avg
    
    # bipartition_0[bipartition_0==True] = torch.tensor(~ bipartition_1).bool()
    # bipartition_0 = bipartition_0.numpy()
    
    bipartition = bipartition.reshape(dims).astype(float)
    # predict BBox
    pred, _, objects,cc = detect_box(bipartition, seed, dims, scales=scales, initial_im_size=init_image_size[1:]) ## We only extract the principal object BBox
    mask = np.zeros(dims)
    mask[cc[0],cc[1]] = 1
    # ipdb.set_trace()
    return np.asarray(pred), objects, mask, seed, None,bipartition
    #(dims[0]*dims[1],dims[0]*dims[1]),second_smallest_vec.reshape(dims[0],dims[1])

def detect_box(bipartition, seed,  dims, initial_im_size=None, scales=None, principle_object=True):
    """
    Extract a box corresponding to the seed patch. Among connected components extract from the affinity matrix, select the one corresponding to the seed patch.
    """
    w_featmap, h_featmap = dims
    objects, num_objects = ndimage.label(bipartition) 
    cc = objects[np.unravel_index(seed, dims)]
    

    if principle_object:
        mask = np.where(objects == cc)
       # Add +1 because excluded max
        ymin, ymax = min(mask[0]), max(mask[0]) + 1
        xmin, xmax = min(mask[1]), max(mask[1]) + 1
        # Rescale to image size
        r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
        r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax
        pred = [r_xmin, r_ymin, r_xmax, r_ymax]
         
        # Check not out of image size (used when padding)
        if initial_im_size:
            pred[2] = min(pred[2], initial_im_size[1])
            pred[3] = min(pred[3], initial_im_size[0])
        
        # Coordinate predictions for the feature space
        # Axis different then in image space
        pred_feats = [ymin, xmin, ymax, xmax]

        return pred, pred_feats, objects, mask
    else:
        raise NotImplementedError

