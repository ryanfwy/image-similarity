'''CLI utility for image preparation.'''

import os
import argparse
import numpy as np


def process(input_dir, delimiter=',', output_path=None):
    '''Generate a `.csv` file with image paths.'''
    result = [['name', 'path']]
    file_names = os.listdir(input_dir)
    file_names.sort()
    for file_name in file_names:
        file_path = os.path.join(input_dir, file_name)
        result.append([os.path.splitext(file_name)[0], os.path.abspath(file_path)])

    if output_path is None:
        parent_dir = list(filter(lambda x: not x == '', input_dir.split('/')))[-1]
        output_path = parent_dir + '.csv'

    np.savetxt(output_path, result, delimiter=delimiter, fmt='%s', encoding='utf-8')

    print('File saved to `%s`.' % output_path)

def main():
    '''CLI entrance.'''
    parser = argparse.ArgumentParser(prog='image_util_cli')
    parser.add_argument('source', action='store', type=str, help='directory of the source images')
    parser.add_argument('-d', '--delimiter', required=False, type=str, default=',', help="delimiter to the output file, default: ','")
    parser.add_argument('-o', '--out-path', required=False, type=str, help='path to the output file, default: name of the source directory')

    args = parser.parse_args()
    if args.source:
        if os.path.isdir(args.source) is False:
            exit('No directory `%s`.' % args.source)

        process(args.source, delimiter=args.delimiter, output_path=args.out_path)


if __name__ == '__main__':
    main()
