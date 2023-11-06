![FakeNews](https://miro.medium.com/v2/resize:fit:640/1*sxSDD2sCznnxRMcF-PARdw.gif)

# _**Fake News Classification Using Navie Bayes, LSTM**_
Fake news classification using Naive Bayes and LSTM involves leveraging two different machine learning approaches to distinguish between real and fabricated news articles. Naive Bayes is a probabilistic algorithm that relies on the assumption of independence between features, making it a straightforward and efficient choice for text classification. It analyzes the words and phrases within news articles to estimate the probability of an article being fake or genuine based on its content. In the new age of technology, we go through lot of news and there is a possibility that the news could be fake. We trained a model which will help us to classify whether the news is genuine or fake. We trained the model using dataset available from Kaggle. This is a binary classification problem and we used LSTM to train the model. The model achieved good accuracy and can be worked upon to even improve it and use it in real-time.

# _**Base Paper**_
+ https://www.researchgate.net/publication/339022255_A_smart_System_for_Fake_News_Detection_Using_Machine_Learning

# _**Algorithm Description**_

**LSTM**:

If you want to predict the next word for the sentence “I like numbers, my favorite subject is …” your answer would be “mathematics”. How do you come to that conclusion, it is because of the word “numbers”. Recurrent Neural Networks help us achieve this. They are neural networks with loops, allowing the information to stay for some time.

![LSTM](https://user-images.githubusercontent.com/88571564/180368044-dbdde927-4bf3-433f-abe3-21d919b42cb9.png)

**Recurrent Neural Network**:

It's the input, the Network analyzes and throws out output ht. RNNs have performed decently but the problem comes when the sentence is too long and has immense data. E.g. predict the next word for the sentence “I like numbers, my favorite subject is mathematics. My brother likes space and technology, his favorite subject is ….” your answer would be “astronomy”. How do you come to that conclusion, it is because of the word “space and technology”.
But how does the network know that word mathematics is not important now and “space and technology is important now”. Also the problem with RNNs is it cannot remember the entire sequence for long. Long Short Term Networks (LSTMs) are a special kind of RNN which are very much capable of handling long term dependencies. LSTMs has a cell state which holds the important word/information required for processing. The information in the cell state can be forgotten (removed) or added based on gates. It has 3 gates to help it decide:
1.	Forget gate: To forget the information in the cell state. Done with the help of a sigmoid function which returns a value between 0 and 1. If 1 or closer to 1, remove the word else retain the word.  
2.	Input gate: What input should be given to the cell state done with the help of tan-h function. 
3.	Update gate: Any information that can be added to the cell state. E.g. : When “space” was in cell state, update it with the word “technology” because word “technology” is important in decision making as well .

![RNN](https://user-images.githubusercontent.com/88571564/180368063-78326934-a189-4f86-a0c5-b07b5f69c0ea.png)

**Reference**

+ https://colah.github.io/posts/2015-08-Understanding-LSTMs/


# _**How to Execute?**_
So, before execution we have some pre-requisites that we need to download or install i.e., anaconda environment, python and a code editor.
**Anaconda**: Anaconda is like a package of libraries and offers a great deal of information which allows a data engineer to create multiple environments and install required libraries easy and neat.

**Download link:**

![Anaconda](https://1.bp.blogspot.com/-UJ1Ws2zZ9V4/TtMbG2ynJiI/AAAAAAAABbM/m6t2kuEhKdY/s1600/The-biggest-anaconda-snake-3.jpg)

https://www.anaconda.com/

**Python**: Python is a most popular interpreter programming language, which is used in almost every field. Its syntax is very similar to English language and even children and learning it nowadays, due to its readability and easy syntax and large community of users to help you whenever you face any issues.

**Download link:**

![Python](https://i0.wp.com/reptileworldfacts.com/wp-content/uploads/2019/05/male-blonde-super-tiger-reticulated-python.jpg?resize=351%2C351&ssl=1)

https://www.python.org/downloads/

**Code editor**: Code editor is like a notepad for a programming language which allows user to write, run and execute program which we have written. Along with these some code editors also allows us to debug, which usually allows users to execute the code line by line and allows them to see where and how to solve the errors. But I personally feel visual code is very good to work with any programming language and makes a great deal of attachment with user.

**Download links:**

![Vs code](https://schwabencode.com/contents/logos/VS2019-Badge.png) ![Pycharm](https://i0.wp.com/scracked.com/wp-content/uploads/2020/01/PyCharm-2019.3.4-Crack.png?fit=200%2C200&ssl=1)

+ https://code.visualstudio.com/Download, 
+ https://www.jetbrains.com/pycharm/download/#section=windows

# _**How to create a new environment and configure jupyter notebook with it.**_
Let us define an environment and why we need different environments. An environment is a collection of libraries that are required to run our project. When we already have an environment with the necessary libraries, why do we need a new environment?
To avoid version mismatches, we create a new environment for each project. For example, in your previous project, you used "tf env" with tensorflow 2.4 and keras 2.4, but in your current project, you must use tensorflow 2.6 and keras 2.6. If you continue your project in the "tf env" environment, there will be a version mismatch and you will need to update tensorflow and keras, but this will cause problems with the previous project's execution. To avoid this, we create a new environment with tensorflow 2.6 and keras 2.6 and resume our project.

Let us now see how to create an environment in anaconda.
+ Type “conda create –n <<name_of_your_env>>”
example: conda create -n env
+ It will ask to proceed with the environment location, type ‘y’ and press enter.
+ When you press ‘y’, the environment will be created. To activate your environment type conda activate <<your_env_name>> . E.g., conda activate myenv.
+ You can see that the environment got changed after conda activate myenv line. It changed from “base” to “myenv” which means you are now working in “myenv” environment.
+ To install a library in your virtual environment type pip install <library_name>.
e.g., pip install pandas
+ Instead of installing libraries one by one you can even install by bunch, i.e., we have a txt file called requirements.tx which consists of all the libraries required to proceed with the project, so we can use it.
+ so, before installing requirements.txt, make sure you are in the specific path where your requirements.txt is located, basically this file is located in the folder where our executable files are located, so we need to move to that directory by following command.
**cd C:\folder_name**
+ Here A -> drive, folder name -> path where your executable file is saved
+ I go to that file path in anaconda using cd command 
1.	Go to drive where your project file is.
2.	Go to the path of your project using cd <path>
3.	Type pip install –r requirements.txt 
+ And all your required libraries will be downloaded and you can start your project.
+ But if you want to use jupyter notebook on the new environment you have to set it up for the new environment.
+ After you have installed all the libraries and created an environment, you need an editor to run the code, that is starting jupyter notebook, as soon as you enter jupyter notebook in the terminal you will definitely get this error. “Jupiter” is not recognized as an internal or external command.
So, to solve it it we have 2 commands.
1.	conda install –c conda-forge jupyterlab
2.	conda install –c anaconda python
Now you are ready to use jupyter on this environment and start with your project!

![thanks](https://media1.tenor.com/images/11ae4fcfc41bb9e66a0176fcfc38e695/tenor.gif?itemid=8486985)
  
  
# _**Steps to execute**_
**Note:** Make sure you have added path while installing the software’s.

1.	Install the prerequisites/software’s required to execute the code.
2.	Press windows key and type in anaconda prompt a terminal opens up.
3.	Before executing the code, we need to create a specific environment which allows us to install the required libraries necessary for our project.
•	Type conda create -name “env_name”, e.g.: conda create -name project_1
•	Type conda activate “env_name, e.g.: conda activate project_1
4.	Make sure you are in the correct path in your terminal, where you have saved your executable file/folder. E.g.: cd A:\project\AI\Completed\project_name, then press enter.
5.	Install necessary libraries from requirements.txt file provided.
6.	Run pip install -r requirements.txt or conda install requirements.txt (Requirements.txt is a text file consisting of all the necessary libraries required for executing this python file. If it gives any error while installing libraries, you might need to install them individually.)
7.	Run Fake News Classification Using Navie Bayes, LSTM.ipynb and make sure to change the path where your executable files are located and also the model path in the code, please follow the link on how to install and set up anaconda environment to execute files.

# _**Data Description**_
Download the Train.csv from link mentioned in Note Section(Down Below)
The Dataset used in this project was collected form the Kaggle data repository which is basically a hub for data scientists and all the enthusiasts who wants to do some exciting things with data. The dataset is basically divided into 2 files, Train & Test. The data was already labelled and segregated into two Files containing alot of data into it.In this Format:

train.csv: A full training dataset with the following attributes:
•	id: unique id for a news article
•	title: the title of a news article
•	author: author of the news article
•	text: the text of the article; could be incomplete
•	label: a label that marks the article as potentially unreliable
•	1: unreliable
•	0: reliable


 # _**Issues Faced.**_
1. We might face an issue while installing specific libraries.
2. Make sure you have the latest version of python or 3.8, since sometimes it might cause version mismatch.
3. Adding path to environment variables in order to run python files and anaconda environment in code editor, specifically in visual studio code.
4. While Downloading Stopwords File From Nltk Library You may refer to Stackoverflow Website.

# _**Note:**_
**All the required data hasn't been provided over here. Please feel free to contact me for any issues. You can also download the dataset from the given link below.**

+ https://www.kaggle.com/c/fake-news

**Credits to the owner who provided the dataset**

### _**Let’s Connect**_
https://www.linkedin.com/in/mudassiruddin21

![Connect](https://media2.giphy.com/media/l1O6zvqu7O317887HF/source.gif)

# _**Yes, you now have more knowledge than yesterday, Keep Going.**_
![Happy](https://media.giphy.com/media/GK7grZYLG7cs0/giphy.gif)
  