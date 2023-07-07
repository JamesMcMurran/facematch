# facerank
Quick and dirty image sorter by faces.
python3 faceoff.py source/directory/ target/directory/ [--recursive] [--ignore] [--alone][--ignoreOver][--face]

--face - The face you want to rank.
--ignoreOver - it will ignore images over this diff value. Helps filter out junk.
--ignore - This will ignore images with no Face or ones with more than one faces. Helps filter out junk.

``` python3 facerank.py --face ./in/Me.jpg --source ./stable-diffusion-webui/outputs/txt2img-images/2023-07-07 --target ./input/Me1/ --ignore --alone --ignoreOver 0.55 ```

# faceoff
Quick and dirty image sorter by faces.
The script will attempt to sort images by face and create a sub-directory with the face ID as directory name.
The original images will be copied to the new sub-directory to make sure data is not lost.
The script now uses multiprocessing to try to live up the the "quick" part.

Make sure to run the pre_req_install.sh before running the script.

## Usage
python3 faceoff.py source/directory/ target/directory/ [--recursive] [--ignore] [--alone]

### Recursion
Recursion was added to enable processing of trees of directories in one go. By adding --recursive to the command this is enabled for the runtime of the script and will treat the source directory as the root of the tree. The target directory will NOT reflect the source tree, it will still be one directory with one sub-directory for every matched face + one folder for images where no faces were detected by the algorithm.

## Ignore faceless images
Add --ignore to simply not copy images where no faces were detected

## Perform "standalone" runs
During my testing I frequently deleted the pickle files up until I got sick of it and added the --alone switch.
This will simply not load face information from previous runs and no data will be stored from the run either.

## Credits
Credit goes to Adam Geitgey for the face_recognition python module.
https://github.com/ageitgey/face_recognition

## Caveats
~~The script does not perform recursion within directories.~~
~~The source directory should only contain image files, no sub-directories.~~
I have not tested with images with several faces, the expected behaviour should be that the same image will end up in two directories.

As stated in the caveats over at https://github.com/ageitgey/face_recognition/blob/master/README.md

* The face recognition model is trained on adults and does not work very well on children. It tends to mix
  up children quite easy using the default comparison threshold of 0.6.
* Accuracy may vary between ethnic groups. Please see [this wiki page](https://github.com/ageitgey/face_recognition/wiki/Face-Recognition-Accuracy-Problems#question-face-recognition-works-well-with-european-individuals-but-overall-accuracy-is-lower-with-asian-individuals) for more details.
