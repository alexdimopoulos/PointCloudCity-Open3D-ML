# Create Point Cloud City SemanticKITTI 
import numpy as np
import sys
import argparse
import psutil
import os
from os import walk
import laspy 
from laspy import convert

vel = 'velodyne'
lab = 'labels'
save_pb = '000000.bin'
save_pl = '000000.label'

def main():
    parser = argparse.ArgumentParser("PCC to SemanticKITTI Format Code", description="Transforms a folder of the Unified Point Cloud City to binary and the SemanticKITTI format")
    parser.add_argument("input_file")
    parser.add_argument("output_dir")

    args = parser.parse_args()

    uni_pcc = sys.argv[1]
    pcc_skitti = sys.argv[2]

    if not os.path.exists(pcc_skitti):
            os.makedirs(pcc_skitti)

    pcc_sk = os.listdir(pcc_skitti)
    las_files = os.listdir(uni_pcc)

    for idx, lfile in enumerate(las_files):
            print(idx, lfile)

            # Create sequence
            s_path = str(idx)
            seq_p = os.path.join(pcc_skitti, s_path)

            # Save to dir
            bin_p = os.path.join(seq_p, vel)
            lab_p = os.path.join(seq_p, lab)
            if not os.path.exists(bin_p):
                    os.makedirs(bin_p)
            if not os.path.exists(lab_p):
                    os.makedirs(lab_p)

            # Save files
            b_p = os.path.join(bin_p, save_pb)
            l_p = os.path.join(lab_p, save_pl)
            
            # #Transform
            las_p = os.path.join(uni_pcc, lfile)
            las = laspy.read(las_p)
            x = las.x 
            y = las.y
            z = las.z
            r = np.zeros(x.shape)
            l = las.classification

            xyz = np.array([x,y,z,r]).transpose()
            l = np.array([l]).transpose()

            xyz.astype('float32').tofile(b_p)
            l.astype('uint32').tofile(l_p)

            print('RAM memory % used:', psutil.virtual_memory()[2])

            del x
            del y
            del z
            del r

            del xyz
            del l

            del s_path
            del b_p
            del l_p
                
            # Getting % usage of virtual_memory ( 3rd field)
            print('Cleared RAM memory % used:', psutil.virtual_memory()[2])

    print('DONE_____')


if __name__ == '__main__':
    main()
