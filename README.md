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

## Example dataset creation

The first step of extractoin is to identify candidates for speech bubbles, for example:

![Example img](./pics/example_page.jpg =400x)

Class: no_text
![Example_no_text](./pics/example_no_text.png =200x)

Class: vert_text
![Example_vert_text](./pics/example_no_vert.png =200x)

Class: hor_text
![Example_hor_text](./pics/example_no_hor.png =200x)

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
