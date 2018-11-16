"""
Semester Thesis: Implementation of the Motion-blur Aware Camera-Pose Estimation Algorithm in Python via Mesh Rendering
Author: Johann Diep (jdiep@student.ethz.ch)

"""

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-io', '--filename_obj', type=str)
    parser.add_argument('-ir', '--filename_ref', type=str)
    parser.add_argument('-or', '--filename_output', type=str)
    parser.add_argument('-mr', '--make_reference_image', type=int, default=1)
    parser.add_argument('-g', '--gpu', type=int, default=1)
    args = parser.parse_args()

if __name__ == '__main__':
    main()
