Reel Data Instructions for Install

1. Download [this GitHub repository](https://github.com/Coltonc18/ReelData) as a .zip file (code > download zip)
   - This repository was private until 5/16/23 at 9pm, at which time it was made publicly available for grading access purposes. The code within was only contributed to by Nathan and Colton, and will be made private as soon as grading for this project is completed.
2. Unpack the zip into a new directory
3. Download and extract the data files from the [Google Drive](https://drive.google.com/drive/folders/11QQ5HrvlVZ10AKYAUC6a7qK4UDKK-j34?usp=share_link) and add to a subfolder named “data”
   - Ensure there is an empty directory called “tests” within the data folder
4. In the root directory, create empty folders named “accuracy\_graphs” and “graphs” if they do not exist
5. Open the Command Prompt from within the CSE163 Anaconda Environment
6. Run the command:conda install -c conda-forge altair vega\_datasets altair\_viewer vega to install the additional libraries used in this project
7. Open VS Code from the same cse163 environment and open the root directory
8. main.py houses a “top to bottom” run of the code for this project
9. graphing\_notebook.ipynb is a Jupyter Notebook which can *usually* display the graphs produced in an easier manner

Directory should look like:

![](dependencies/Aspose.Words.2f31ee54-6578-4362-bc6d-f9815cbbbe8d.001.png)
