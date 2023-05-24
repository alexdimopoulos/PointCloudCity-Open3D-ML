# Create PCC SemanticKITTI (PCC_SKITTI) from PSCR's Point Cloud City
import argparse
import sys
import os
import numpy as np
import laspy

from laspy import convert

uni_codes = {
    0:	0,
    1: 0,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 12,
    13: 13,
    14: 14,
    15: 15,
    16: 16,
    17: 17,
    18: 18,
    19: 19,

    24:	1,
    104: 1,
    68: 2,
    73: 3,
    78: 4,
    112: 5,
    25: 6,
    96: 6,
    100: 7,
    75: 8,
    77: 8,
    103: 9,
    81: 10,
    76: 11,
    92: 12,
    28: 13,
    64: 13,
    82: 14,
    105: 15,
    22: 16,
    109: 16,
    23: 17,
    74: 17,
    80: 18,
    21: 19,
    110: 19
}


def format_pcc_skitti(las_paths, out_path):
    vel = 'velodyne'
    lab = 'labels'
    save_pb = '000000.bin'
    save_pl = '000000.label'

    # Add semantic KITTI structure
    outp_path = os.path.join(out_path, 'PCC_SKITTI')
    outpu_path = os.path.join(outp_path, 'dataset')
    output_path = os.path.join(outpu_path, 'sequences')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for indx, l_path in enumerate(las_paths):
        seq_count = str(indx).zfill(2)
        seq_p = os.path.join(output_path, seq_count)
        if not os.path.exists(seq_p):
            os.makedirs(seq_p)

        # create doc and save building name and path
        building_name = pc_name = os.path.basename(l_path)
        pc_name = pc_name[:-4] + '.txt'
        doc_p = os.path.join(seq_p, pc_name)
        try:
            with open(doc_p, 'w') as f:
                f.write(l_path)
        except FileNotFoundError:
            print("The building name directory not initialized")

        # Set up save paths 
        bin_p = os.path.join(seq_p, vel)
        lab_p = os.path.join(seq_p, lab)

        if not os.path.exists(bin_p):
            os.makedirs(bin_p)
        if not os.path.exists(lab_p):
            os.makedirs(lab_p)

        b_p = os.path.join(bin_p, save_pb)
        l_p = os.path.join(lab_p, save_pl)
        
        # returns laspy las point cloud
        las = unify_las_pcc(l_path)
        x = las.x 
        y = las.y
        z = las.z
        r = np.zeros(x.shape)
        l = las.classification

        xyz = np.array([x,y,z,r]).transpose()
        l = np.array([l]).transpose()

        print('saving file...')
        xyz.astype('float32').tofile(b_p)
        l.astype('uint32').tofile(l_p)
        print('SAVED')

        del x
        del y
        del z
        del r
        del xyz
        del l
        del las
    return

def unify_las_pcc(pc):
    # Change las file class labels
    pc_name = os.path.basename(pc)
    pc_name = pc_name[:-4]
    print(f'Unifying PC: {pc_name}')
    file_size = os.path.getsize(pc)
    print("File Size is :", file_size, "bytes")
    las = laspy.read(pc)
    las = convert(las, point_format_id=7)
    for i, class_label in enumerate(las.classification):
        if class_label in uni_codes:
            las.classification[i] = uni_codes[class_label]
        else: las.classification[i] = 0
    return las

def main():
    parser = argparse.ArgumentParser("PCC Unification Code", description="Unifies the point cloud labels of Poaint Cloud City LAS files")
    parser.add_argument("input_file")
    parser.add_argument("output_dir")
    args = parser.parse_args()

    pcc_path = sys.argv[1]
    output_path = sys.argv[2]

    print('Input dir: ' + pcc_path)
    print('Output dir: ' + output_path)

    pcfiles = []
    for dirpath, subdirs, files in os.walk(pcc_path):
        for x in files:
            if x.endswith(".las"):
                pcfiles.append(os.path.join(dirpath, x))
    print(pcfiles)
    print(len(pcfiles))
    format_pcc_skitti(pcfiles, output_path)

if __name__ == '__main__':
    main()




