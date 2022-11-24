'''
    File name:
    Author: Xu Liu and https://github.com/strawlab/pymvg
    Date created:
    Date last modified:
    Python Version: 2.7
'''

from __future__ import (print_function, division, absolute_import)
import numpy as np


def trunk_corners_multi_view_triangulation(K_mat, rot_trans_dict,trunk_tracks):
    """Find 3D coordinate using all data given
    Implements a linear triangulation method to find a 3D
    point. For example, see Hartley & Zisserman section 12.2
    (p.312).
    By default, this function will undistort 2D points before
    finding a 3D point.
    """
    trunk_corners_3D_pos = []
    for trunk_track in trunk_tracks:
        A=[]
        for frame_name, pos in trunk_track.hist_fr_idx_dict.items():
            frame_idx_str = str(int((frame_name.split('frame'))[-1]))
            if frame_idx_str not in rot_trans_dict:
                continue
            rot_mat = rot_trans_dict[frame_idx_str][0]
            trans_vec = rot_trans_dict[frame_idx_str][1].reshape(3,1)
            rot_trans_mat = np.append(rot_mat, trans_vec, 1)
            Pmat = np.dot(K_mat, rot_trans_mat) # Pmat is 3 rows x 4 columns
            row2 = Pmat[2,:]
            x,y = pos
            A.append( x*row2 - Pmat[0,:] )
            A.append( y*row2 - Pmat[1,:] )
        # Calculate best point
        A=np.array(A)
        u,d,vt=np.linalg.svd(A)
        X = vt[-1,0:3]/vt[-1,3] # normalize
        trunk_corners_3D_pos.append(X)
        print(X)
    return trunk_corners_3D_pos
