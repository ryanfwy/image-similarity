# Image Similarity

This is an efficient utility of image similarity using [MobileNet](https://arxiv.org/abs/1704.04861) deep neural network.

Image similarity is a task mostly about feature selection of the image. Here, the Convolutional Neural Network (CNN) is used to extract features of these images. It is a better way for computer to understand them effectively.

This repository use a light-weight model, the MobileNet, to extract image features, then calculate their cosine distances as matrixes. The distance of two features will lie in `[-1, 1]`, where `-1` denotes the features are the most unlike, and `1` denotes they are the most similar. Choose a proper threshold `[-1, 1]`, the most similar images will be matched.

## Usage

The code is written to match the similar images in a huge amount as efficiently as possible.

To use it, two `.csv` source files should be prepared before running. Here is an example of one source file. By default, the `.csv` file should at least include one field that place the urls [[1]](#notice).

```text
id,url
1,https://raw.githubusercontent.com/ryanfwy/image-similarity/master/demo/1.jpg
2,https://raw.githubusercontent.com/ryanfwy/image-similarity/master/demo/2.jpg
3,https://raw.githubusercontent.com/ryanfwy/image-similarity/master/demo/3.jpg
4,https://raw.githubusercontent.com/ryanfwy/image-similarity/master/demo/4.jpg
5,https://raw.githubusercontent.com/ryanfwy/image-similarity/master/demo/5.jpg
6,https://raw.githubusercontent.com/ryanfwy/image-similarity/master/demo/6.jpg
```

After that, we can setup the number of processes that are used to request images from the urls parallelly. For example, we use 2 processes with this tiny demo.

```python
similarity.num_processes = 2
```

For feature extraction, a data generator is used to predict images with model batch by batch. By default, GPU will be used if it satisfy the conditions of [Tensorflow](https://www.tensorflow.org/install/gpu). Now we can set a proper size of batch based on the memory size of our computer or server. In this demo, we set it to 16.

```python
similarity.batch_size = 16
```

After invoking the function `save_data()` two times, four self-generated files will be saved into `__generated__` directory with the file names of `_*_feature.h5` and `_*_fields.csv`. We can further calculate the similarities by calling `iteration()`, or load the generated files at any time afterward.

Totally, the full example will look like:

```python
similarity = ImageSimilarity()

'''Setup'''
similarity.batch_size = 16
similarity.num_processes = 2

'''Load source data'''
test1 = similarity.load_data_csv('./demo/test1.csv', delimiter=',')
test2 = similarity.load_data_csv('./demo/test2.csv', delimiter=',', cols=['id', 'url'])

'''Save features and fields'''
similarity.save_data('test1', test1)
similarity.save_data('test2', test2)

'''Calculate similarities'''
result = similarity.iteration(['test1_id', 'test1_url', 'test2_id', 'test2_url'], thresh=0.845)
print('Row for source file 1, and column for source file 2.')
print(result)
```

or if the files have been generated before:

```python
similarity = ImageSimilarity()
similarity.iteration(['test1_id', 'test1_url', 'test2_id', 'test2_id'], thresh=0.845, title1='test1', title2='test2')
```

For practical usage, the `thresh` argument of `save_data()` is recommended to be in `[0.84, 1)`. One balanced value can be `0.845`.

Any other details, please check the usages of each function given by `main_multi.py`.

## Requirements and Installation

**NOTE**: Tensorflow is not included in `requirements.txt` due to the platform differences, please install and configure yourself based on your computer or server. Also note that `Python 3` is required.

```pip
$   git clone https://github.com/ryanfwy/image-similarity.git
$   cd image-similarity
$   pip3 install -r requirements.txt
```

The requirements are also listed down bellow.

- tensorflow: the newest version for CPU, or the version that matches your GPU and CUDA.
- h5py~=2.6.0
- numpy~=1.14.5
- requests~=2.21.0

## Experiment

In the demo, 6 and 3 images are used to match their similarities.

### Accuracy

The cosine distances are shown in the table.

| | <img width="100" src="./demo/3.jpg"/> | <img width="100" src="./demo/4.jpg"/> | <img width="100" src="./demo/5.jpg"/> |
| --- | :---: | :---: | :---: |
| <img width="100" src="./demo/1.jpg"/> | **0.9229318** | 0.5577963 | 0.5826051 |
| <img width="100" src="./demo/2.jpg"/> | **0.84877944** | 0.538753 | 0.5624183 |
| <img width="100" src="./demo/3.jpg"/> | **1.** | 0.5512465 | 0.57025677 |
| <img width="100" src="./demo/4.jpg"/> | 0.5512465 | **0.99999994** | 0.54037786 |
| <img width="100" src="./demo/5.jpg"/> | 0.57025677 | 0.54037786 | **0.9999998** |
| <img width="100" src="./demo/6.jpg"/> | 0.5575757 | 0.5238174 | **0.91234696** |

As it is shown, image similarity using deep neural network works fine. The distances of the matched images will roughly be greater than `0.84`.

### Efficiency

For running efficiency, multi-processing and batch-wise prediction are used in feature extraction procedure. And thus, image requesting and processing in CPU, image prediction with model in GPU, will run simultaneously. In the procedure of similarity analysis, a matrix-wise mathematical method is used to avoid n*m iteration one by one. This may help a lot in the condition of low efficiency of python iteration, especially in a huge amount.

Table bellow shows the time consumption runing with 8 processes in a practical case. The results are only for reference, they may change a lot based on the number of processes we use, the quality of the network, the image size of the online resources and so on.

|  | Source 1 | Source 2 | Iteration |
| :---: | :---: | :---: | :---: |
| Amount | 13501 | 21221 | 13501 * 21221 |
| Time Consumption | 0:35:53 | 0:17:50 | 0:00:03.913282 |

## Notice

[1] By default, the programme have to get the online images from urls we prepared in `.csv`. If we want to run the code with a list of offline images, we need to override the `_sub_process()` class method by ourselves. For demo and details, please check [demo_override](./demo_override).


## Thanks

Demo images come from [ImageSimilarity](https://github.com/nivance/image-similarity) by [nivance](https://github.com/nivance). It is an another algorithm (pHash) of image similarity implementation in java.
