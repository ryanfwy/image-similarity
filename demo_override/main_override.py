import sys
sys.path.append('..')

from main_multi import ImageSimilarity, DeepModel

class NewImageSimilarity(ImageSimilarity):
    @staticmethod
    def _sub_process(para):
        # Override the method from the base class
        path, fields = para['path'], para['fields']
        try:
            feature = DeepModel.preprocess_image(path)
            return feature, fields

        except Exception as e:
            print('Error file %s: %s' % (fields[0], e))

        return None, None


if __name__ == "__main__":
    similarity = NewImageSimilarity()

    '''Setup'''
    similarity.batch_size = 16
    similarity.num_processes = 2

    '''Save features and fields'''
    similarity.save_data('./test1.csv', ',')
    similarity.save_data('./test2.csv', ',', cols=['id', 'path'])
