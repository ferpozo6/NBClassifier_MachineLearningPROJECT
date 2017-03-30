---
## ADVANCED GENOMICS BIOINFORMATICS 
# Naive Bayes Assignment
# MODELING BRAIN REGIONS WITH A PREDICTIVE NAIVE BAYES MODEL

**Marcos Camara Donoso**
marcos.camara01@estudiant.upf.edu
**Fernando Pozo Ocampo**
fernando.pozo01@estudiant.upf.edu
February, 2017

USAGE : $sh NaiveBayes_BrainCamaraMarcos_PozoFernando.py (function in the last line -> Naive_Bayes_Brain (training set (.txt),  testing set(.txt), input file (.psi))) 
---
This program try to predict the brain region of your brain sample. 
The brain region samples have been sequenced by RNA-seq, consequently annotated and lastly converted in a psi file by SUPPA (Software which generates the different alternative splicing events from an input annotation file, quantifying the event inclusion levels (PSIs)).

It proposes a predictive model with selected 650 samples well balanced and chosen by region brain and human genre for the training dataset. You can test the performance with my test dataset which contains 7 samples of each brain or you can choose you own samples in order to predict the class of each of them. Even, you can try to improve the model introducing another quantity of region brain samples.

If you want help to create the training and testing sets, please open *parseGTEX.pl*.

Firstly, the script creates a normalized confusion matrix with the score of your predictions.

Closing the figure, the program continues and inmediately prints the accuracy results of the predictive model, divided by class (each region brain) and overall statistics.
Finally, it also returns a .csv tab delimited output:
*Score Prediction  Label   Sample identifier*

*Please, don't forget to check the right installation of the imported modules in order to run this script properly. It is recommended to install Anaconda distribution of Python, aiming to simplyfing the package management and deployment.*
