# Classifying fruits on Jetson Nano

Classification of fruits with transfer learning using Tensorflow on Jetson Nano.

Before proceeding, make and mount a swap file (at least 32GB). This step is mandatory. More details [here](https://support.rackspace.com/how-to/create-a-linux-swap-file/).

```bash
sudo fallocate -l 32G /mnt/32GB.swap
sudo chmod 600 /mnt/32GB.swap
sudo mkswap /mnt/32GB.swap
sudo swapon /mnt/32GB.swap
```

For a permanent configuration, add the following line at the end of `/etc/fstab` :

```bash
# Open the /etc/fstab file in an editor with administrative privileges
sudo nano /etc/fstab
```

```bash
# Add the following line to the end of the file
/mnt/32GB.swap  none  swap  sw 0  0
```

Run `sudo tegrastats` to confirm that the swap file is mounted.

Check if [Tensorflow for Nvidia Jetson Nano](https://forums.developer.nvidia.com/t/official-tensorflow-for-jetson-nano/71770) is already installed:

```bash
python3 -c "import tensorflow as tf; print(tf.__version__)"
```

If the previous command did not output `2.7.0` please install [Tensorflow for Nvidia Jetson Nano](https://forums.developer.nvidia.com/t/official-tensorflow-for-jetson-nano/71770):  

```bash
sudo apt-get update
sudo apt-get install -y python3-pip pkg-config
sudo apt-get install -y libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
sudo ln -s /usr/include/locale.h /usr/include/xlocale.h
sudo pip3 install --verbose 'protobuf<4' 'Cython<3'
sudo wget --no-check-certificate https://developer.download.nvidia.com/compute/redist/jp/v461/tensorflow/tensorflow-2.7.0+nv22.1-cp36-cp36m-linux_aarch64.whl
sudo pip3 install --verbose tensorflow-2.7.0+nv22.1-cp36-cp36m-linux_aarch64.whl
sudo pip3 install tensorflow-hub keras==2.6.0
```

First check if the `jetson-fruits-classification` project exists in `repos` folder from your `home` directory.

If the project does not exists clone the project repository with:

```bash
git clone https://github.com/Reddimus/jetson-fruits-classification
cd jetson-fruits-classification
```

For classification, a proper dataset is required. The available datasets (FIDS30 and Fruits360) did not meet the requirements, so a new dataset with 6 classes and a total of 600 images (100 per class) was created using the `camera-capture` utility from the Hello AI World example.

The dataset is available in the *fruits-dataset* folder of the repository; alternatively, another dataset may be used or a new one created.

The classes in the dataset are:

```txt
• Apple/
• Beetroot/
• Dates/
• Mango/
• Orange/
• Pomegranate/
```

## Inference

Test the model using fruit images. Some test images from Google Images are provided in the *test-images* folder of the repository. Either use the trained model or the pretrained model available in the *model* folder. Ensure that the desired model is copied to the *retrain* folder and run:

```bash
python3 retrain/label_image.py --graph=model/output_graph.pb --labels=model/output_labels.txt --input_layer=Placeholder --output_layer=final_result --image=test-images/apple.jpeg
```

After execution, output similar to the following should be observed:

```txt
apple 0.9856818
orange 0.005912187
dates 0.002886865
pomegranate 0.0026501939
mango 0.0014653547
```

This indicates that the image was correctly classified as an apple. Additional testing with other images is recommended by replacing the test image path.

## Creating a Dataset

Details for using the `camera-capture` utility are available [here](https://github.com/dusty-nv/jetson-infehttps://github.crence/blob/master/docs/pytorch-collect.md).

Create a directory for the new dataset and add a file named *labels.txt* in its root that lists the dataset's class names (one per line):

```bash
mkdir my-fruits-dataset
cd my-fruits-dataset
gedit labels.txt
```

Save the file after adding the class names.

Navigate to `jetson-inference/build/tools/camera-capture` and run:

```bash
make
```

This compiles an executable named `camera-capture` located in `jetson-inference/build/aarch64/bin`. Then, move to that directory and execute:

```bash
cd ../../aarch64/bin/
./camera-capture
```

The utility will launch. Assume the CSI camera on the Jetson Nano is used; for other cameras, refer to the linked documentation. Select the dataset directory and the *labels.txt* file. Once the class names are loaded, images can be captured. Capture images for each class in the 'train' folder only (a separate 'train' and 'val' structure is not required for retraining). Use the spacebar shortcut for quicker image capture, and collect at least 100 images per class to achieve effective retraining results.

After capturing, move all class folders from the 'train' directory to the root of the `my-fruits-dataset` so that the directory structure appears as follows:

```txt
‣ my-fruits-dataset/
  • class-A/
  • class-B/
  • class-C/
  • labels.txt
```

Note that *labels.txt* is not required after this step.

## Re-training using Tensorflow

The following instructions are based on the official Tensorflow tutorial for image retraining, detailed [here](https://www.tensorflow.org/hub/tutorials/image_retraining).

If the pretrained model provided in the repository meets the requirements, this retraining step can be skipped.

For retraining, the process can be executed on either the Jetson Nano or a host PC. Clone the repository and run the retraining script; upon completion, the model is saved as `/tmp/output_graph.pb` and `/tmp/output_labels.txt`.

From the repository root, open a terminal and execute (ensuring the dataset directory is specified correctly):

```bash
python3 retrain/retrain.py --image_dir fruits-dataset/
```

This initiates the training process. On the Jetson Nano, training may take approximately 1 hour, though a host PC may be used with subsequent model transfer. Ensure that the Tensorflow version is consistent across systems to prevent compatibility issues.

If any module errors occur, install the missing modules using `pip3` and re-run the script.

After retraining, save the two output files from the *tmp* folder since they will be cleared upon system shutdown.
