# AccountAnalyzer
My first ever peace of code for Python Keras DNN that categorizes account transactions

Hello! I'm Jani and I've recently reated my very first AI project and made it public here at GitHub. I've had a two-fold career so far as a SW professional and as a poker player. But recently I've wanted to learn something
totally new that fits my skills. I'm a mathematically oriented MSc and have been seeing and hearing about AI and neural networks for the past 5 years increasingly. During that time I've slowly started to get myself first
familiar with R and now more with Python and neural networks through Andrew Ng's Coursera courses (highly recommended). Last summer I started to organize some my family's account lines into categories to better understand
where does our money come and go. For this I exported a CSV file from my online bank and started to categorize the transactions. Only later I figured this would be a nice idea to try to create my first DNN around.

I have a total of 20 categories that I manually categorized the input data, such as: eating/dringing indoors, eating/drinking outdoors, salary, cloting, health, leisure time, traffic, insuranse etc. From the bank CSV I was
able to get these input vectors: 1) date of the transaction (actually 3 dates, but I tossed the other two as irrelevant to the task), 2) amount in euros that was transacted, with plus or minus sign, 3) receiver/payer
(string), 4) account number (string), 5) transaction description (string) and 6) message, if any (string).

I started with 2-hidden layer neural network and selected layer-1 size to be 12 hidden units and layer-2 to be 8 hidden units as the amount on the latter layers usually seem to decrease in the DNNs I've seen. I didn't have
any way to input the string-types yet so as a SW developer I took the easy way out and transformed the strings to int by using their length. :) Better than nothing and so it seemed. I had a bit more that 1600 lines of
transactions that cover our family's transactions through a year in calendar time. I did learn forward and backward propagation on Andrew Ng's cources and how to do gradient decent, but I still had wanted to learn more
on Tensorflow and Keras that had been briefly introduced on the course so those were my first package choises as it seemed like the modern way to approach my "Hello World" of a sort.

So for reading the input, I used Pandas read_csv, for other techs I used are Keras, Tensorflow, NumPy and Sklearn. I tried different activation methods, but ended up using relu in hidden layers and softmax in the output
layer as I wanted only a single category out of 20 to be selected in the end for each line. I had read good things about Adam optimizer so that and RMSprop were the selected few that I evaluated with different learning
rates. Both seemed to fare almost equally in my task in the end.

I prepared my input data by normalizing it so that my weights would not head to infinity (+/-) or zero too easily. I took the maximums of the feature vector and divided the values with it to get values in the range [-1, 1].
So the first trials after getting all the Python code in good shape has some light at the end of the tunnel: Accuracy (mean of n_split runs on the data) was 27.33% when running 500 epochs and batch size = n. I started to
search semi randomly on the parameters and increased the number of epochs so that the longest runs on my CPU lasted about 4 hours through the night. Result accuracies was as low as 4.67% and as high as 65.29%!!! So that
means almost 2/3 of my DNNs guesses got the right category in its "guessing" on what category the given transaction belongs to. However this seemed to be the maximum that I could receive with even deeper networks with 4
hidden layers and various layer sizes. So it was back to the drawing board.

I listened to a machine learning podcast about NLP and figured there was a better way to convert strings to numbers than what I had selected. Several of course, but I decided to use CountVectorizer from
sklearn.feature_extraction.text to create a dictionary of words from each feature vector. This was presented as a binary matrix that increased my features from 6 to 1249! A slight increase I would say, but I just plugged
that number in to my model and off I started to seek for new optimum hyperparameters for my enhanced model. After few more days of tryouts I seemed to come close to the maximum accuracy of 88-89%. This seems good enough
for me now as 8/9 will be selected to the correct category with these several different types of neural networks. Similar results were found with 3-layer DNNs and 2-layer DNNs. One of the smalles DNNs that crossed the 88%
mark was 30 and 25 hidden unit layers with just 100 epochs. This could have had some positive variance to it also so I rely on the deeper ones like 600-150-40 hidden layer units a bit more with similar results.

I also think that it is pretty had to get even better results than this out of this data as it contains so many single typed transactions that the NN has very low possibility to get any grip on how it should be categorized.
Therefore I'm quite happy with this end result and continue to next cources and challenges. Maybe even to a data scientist job one day... :) Thanks for reading!