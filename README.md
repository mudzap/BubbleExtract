# BubbleExtract

Set of python scripts to train, and extract speech bubbles from raw manga.

## Usage

For training, just execute train_type.py with the number of epochs to train for:
```
python train_type.py --epochs 200
```

To extract the speech bubbles from a set of images, take a look at classify.py, it should be fairly intuitive what changes you might need to do.

## Dataset

No dataset was available to me, so I extracted it and made it myself with functions from the get_data.py file, unfortunately, this was not a very good source of data, so the given network might not be performing as well as it should.

## Dependencies

- OpenCV
- Tensorflow (2.4.1+ for training)
- NumPy
- argparse (for training)
- sklearn (for training)


## CNN Architecture

The CNN architecture is simple:

- 2 data augmentation layers (RandomZoom and RandomTranslation)
- 1, 32 7x7 kernel size convolutional layer
- 1, max pooling layer
- 1 dropout layer
- 1 flatten layer
- 1, input neuronal layer with 32 neurons
- 1, hidden neuronal layer with 24 neurons
- 1, output neuronal layer

## Further improvements

Integration with tesseract, automatic preprocessing of extracted speech bubbles and training with a more 'respectable' dataset should pave the way for identifying possible improvements, so I should get started with that.

## License

LGPL 2.1
