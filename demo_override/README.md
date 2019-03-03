# Implement Your Own `_sub_process()`

By default, the `.csv` source file should at least include one field that place the **urls**. In other words, the programme have to get the online images from urls. However, if we want to run the code with a list of offline images, we need to override the `_sub_process()` class method by ourselves.

## Implement the Subclass

The implementation should look like:

```python
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
```

As it is shown, the method `_sub_process()` just simply remove one line `request.get(path)` and pass the `path` argument to `DeepModel.preprocess_image()` directly.

In here, the `.csv` source file should at least include a field, such as `path`, to place all the local image paths. For example, it can be prepared like this.

```
id,path
3,../demo/3.jpg
4,../demo/4.jpg
5,../demo/5.jpg
```

The full example is also given in [main_override.py](./main_override.py). Please read it for more details about how to implement your own `_sub_process()` and run.

## Quick Preparation

If we want to load a batch of offline image paths from the local directory which are prepared for `.csv` source file, the [image_util_cli.py](../image_util_cli.py) quick preparation script can easily do this job.

To run this script, you should first put a batch of images into a directory, such as `source1`. The document tree will look like this.

```
./source1
 |- image1.jpg
 |- image2.jpg
 |- ...
 |_ image100.jpg
```

After that, open `Terminal.app` (MacOS), `cd` to the directory of `image_util_cli.py`, and run it with the required arguments.

```
$   cd image-similarity
$   python3 image_util_cli.py ./source1 -d '\t' -o ./images.csv
```

The usage of `image_util_cli.py` is given bellow. Also we can check it at any time by passing the argument `-h`.

```
usage: image_util_cli [-h] [-d DELIMITER] [-o OUT_PATH] source
positional arguments:
source                directory of the source images

optional arguments:
-h, --help            show this help message and exit
-d DELIMITER, --delimiter DELIMITER
                      delimiter to the output file, default: ','
-o OUT_PATH, --out-path OUT_PATH
                      path to the output file, default: name of the source directory
```
