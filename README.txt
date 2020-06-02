APPROACH

1.I have considered each image plate and applied image processing techniques like binarize the image and morphological operation on that.
2.Later segmenting each number/alphabet from the number plate using connected components algorithm in computer vision.
3.Using Contour detection approximated the position of each number/alphabet in the number plate.
4.I have thought of different methods like extracting the number plate rectangular box and then do four-point transformation on each plate that can improve the accuracy to great extent but for all the number plates I was not able extract the rectangular bounding box.
5.So, I have concentrated on segmenting each number/alphabet in the number plate.
6.Later these numbers/alphabets are pumped to CNN classification algorithm which has 36 classes 0-9 and A-Z.
7.Before joining all the text into single string I have used my own bounding boxes sorting algorithm.
8.I tried to get the dataset for the image and got one but the accuracies are not good at present. I just wanted to let u know that I have collected the data from given dataset itself and it will take some more time for me to improve the accuracy for the model.
9.Manually labelling training data is taking a bit of time.
10.Anyways I have extracted number plate text using the model which I have trained.
11.The data used for training also I am uploading to github you can see them as well.
12.I have tried my best to do the assignment. Please let me know the feedback and the approach you thought of.
13.I have collected all the number plate digits/alphabets and train on them but labelling them will take some time. So, I have trained based on another dataset which I found online.
14.Link to dataset: http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/

