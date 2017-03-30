# 1. Importing libraries as python modules
import numpy as np
import math as mt
import re
import pandas as pd
import matplotlib.pyplot as plt
from pandas_ml import ConfusionMatrix as CM

#2. Defining the main function of my script. I'll call at the end of the script writing the name of the subsequent files used.
def Naive_Bayes_Brain(trainfile, testfile, psifile):

	"""This class predicts the brain region of your sample. It proposes a predictive model with a training dataset, but you can modify this dataset in order to modify the final performance of the script."""
	print("The program has already started, please wait...")

	# 2.1 Defining my list and dictionaries that it will use.
	regions=[]
	ListSam=[]
	values=[]
	list_samplestesting=[]
	RegionSamples_test={}
	RegionSamples_dict={}
	training_dict={}
	testing_dict={}
	testing_final={}

	# 2.2 Opening and parsing the training file
	with open(testfile) as txt:
		for line in txt:
			listest=re.split(r'\t',line.rstrip("\n"))		# Regular expresion for strip the line and save the columns in a variable.
			RegionSamples_test.setdefault(listest[1], []).append(listest[0])	# Creates a dictionary with a list inside

	# 2.3 Opening and parsing the training file
	with open(trainfile) as txt:                                #.txt file with the samples
		for line in txt:
			listed=re.split(r'\t',line.rstrip("\n"))            # Puts the line in a list
			RegionSamples_dict.setdefault(listed[1], []).append(listed[0]) # Creates the dictionary with the Region and add to a list the samples in the region
			if listed[1] not in regions:                        # Creates the list of regions for use later on the script
				regions.append(listed[1])

	psi=open(psifile)                                           # .psi file with the events and values
	IdSamp=psi.readline()                                       # Reads the first line of the file
	ListSamp=re.split(r'\t',IdSamp.rstrip("\n"))                # Puts the identifiers in a list eliminating tabs and new lines
	psi.close()													# Closes the file

	# 2.4 Creating the main for-loop for RegionSamples_dict dictionary created before. Don't forget that it is for training file
	for region, sample in RegionSamples_dict.items():           # For each region and sample in the dictionary

		 # 2.4.1 Opening the file with the samples identifiers (1st line) and 1 event per next lines (37826 tottaly)
		with open(psifile) as psi:                              # Opens the file each time, it has to iterate for it
			next(psi)                                           # Skips the first line
			events_dict={}                                      # Empties the events dictionary

			# 2.4.1.1 For-loop for the bigger file, it takes the splicing events and saves it in a variables
			for event in psi:
				event_values=re.split(r'\t',event.rstrip("\n"))
				values=event_values[1:]                         # Takes the values skiping the identifier of the event
				ups=[]                                          # Reseting the lists for a new event
				downs=[]
				nas=[]

				if (values.count("NA")/len(values)) < 0.1:      # Threshold with the minimum proportion of NA's (non defined PSIs) in the event
					for num in range(len(ListSamp)):            # For each index of Samples
						if ListSamp[num] in sample:             # If a sample of the list is in a region
							if values[num] == "NA":             # If the value for a given sample in a event is NA append it to the NA list
								nas.append(ListSamp[num])
							elif float(values[num]) >= 0.5:     # If the value for a given sample in a event is higher than 0.5 then save the sample in a list of ups for that event
								ups.append(ListSamp[num])
							else:                               # If not higher appends it to a downs list
								downs.append(ListSamp[num])

				## Building the dictionary of samples with ups, downs, and NA for a given event
				events_dict.setdefault(event_values[0],[["up", ups],["down", downs],["NA", nas]])

			# 2.4.1.2 Building the dictionary for the lists of samples with ups, downs and NA for all the events in a region
			training_dict.setdefault(region, events_dict)

	# 2.5 For-loop for starting to creates the RegionSamples_test dictionary for the test file
	for region, sample in RegionSamples_test.items():           # For each region and sample in the dictionary

		# 2.5.1 Opening the PSI file again
		with open(psifile) as psi:                              # Opens the psi file
			next(psi)                                           # Skips the first line

			# 2.5.1.1 For-loop which has the same performance as the started in line 101
			for event in psi:
				event_values=re.split(r'\t',event.rstrip("\n"))
				values=event_values[1:]                         # Take the values skiping the identifier of the event
				ups=[]                                          # Reset of the lists for a new event
				downs=[]
				nas=[]

				if (values.count("NA")/len(values)) < 0.1:      # Threshold of the min proportion of NA's in the event
					for num in range(len(ListSamp)):            # For each index of samples
						if ListSamp[num] in sample:				# If a sample of the list is in a region
							if values[num]!='NA':
								if ListSamp[num] not in testing_dict:
									testing_dict[ListSamp[num]]=[["up",[]], ["down",[]],["NA",[]]]
								if float(values[num])>=0.5:
									testing_dict[ListSamp[num]][0][1].append(event_values[0])
								elif float(values[num])<0.5:
									testing_dict[ListSamp[num]][1][1].append(event_values[0])
							else:
								if ListSamp[num] not in testing_dict:	# Saving in a list the values that doesn't have PSIs (NA)
									testing_dict[ListSamp[num]]=[["up",[]], ["down",[]], ["NA",[]]]
								if values[num] == "NA":
									testing_dict[ListSamp[num]][2][1].append(event_values[0])


	print("You have created the dictionaries succesfully !!!")

	# 2.6 Initializing our next dictionaries. One of them. Here we start to adding the formula for create our Naive Bayes model
	pmydict_littlefile = {}  # It will compute the training set
	pmydict_wholefile = {}	 # It  will compute the lenght of the whole samples

	my_probabilities_dict = {}     ## Final dictionaries
	my_ntotal_dict = {}

	males_and_females=50     ## We are going to take 29 males and 21 females of every brain region in order to randomized the samples in a right way.

	## Training set contains a dictionary with brain region of primary key, splicing event as secondary key and a list of up, downs and NA which contains the subsequent samples for each region with the right value (up or down)
	for key_primary_bregion, value_for_bregion in training_dict.items():  ## = BRAIN REGION, Value_for_bregion = (ENS: [up, GTEX_id], [down, GTEX_id] [NA,  GTEX_id])....

		pmydict_littlefile = {}        # We restart each time the dictionary in order to iterate and make the for loop correctly.
		pmydict_wholefile = {}         # This is the only way to calculate the probabilities.

		for key_secondary_ens, value_for_ens in value_for_bregion.items():    ## Accesing to the dictionary inside the main dictionary created before. key_secondary = ENS, Value_for_ens = ([up, GTEX_id], [down, GTEX_id], [NA,  GTEX_id])....
			psi_total=[]
			psi_likelihood=[]

			# 2.6.1 For-loop which computes the values for psi
			for psi in value_for_ens:
				count_psi = psi[0]          # Counting the total number of up, downs or NA saving in the first element of our list !
				npsi = len(psi[1])          # Total number of possible attributes values. You have to compute the total lenght of this values in order to calculate the likelihood after
				psi_total.append([count_psi, npsi]) ## Creating a list with the number of each PSI value and the total number of each value

				p_for_se = ((npsi+1)/(males_and_females+3)) # The formulae of probability of attribute value (Predictor prior probability). Adding pseudocounts correction in order to compute the right values for next logarithmics calculus.
				psi_likelihood.append([p_for_se])			# Creating the list with the likelihoods of Naive Bayes formula (P(SE|brainregion))

			# 2.6.2 Creating a dictionary with SE events as keys and its counts for up, down and NA	as values
			pmydict_wholefile.setdefault(key_secondary_ens, psi_total)  ## Saving my secondary keys (SE event) and the subsequent values of attributes and total n
			## EXAMPLE :
			## ENSG00000000419.12;SE:chr20:50940933-50941105:50941209-50942031:- [['up', 0], ['down', 45], ['NA', 0]]
			## ENSG00000000460.16;SE:chr1:169806088-169807791:169807929-169821679:+ [['up', 39], ['down', 5], ['NA', 1]]

			# 2.6.3 Creating a dictionary with SE events as keys and its likelihood probabilities  for up, down and NA as values
			pmydict_littlefile.setdefault(key_secondary_ens, psi_likelihood)
			## EXAMPLE :
			#	ENSG00000000460.16;SE:chr1:169806088-169807791:169807929-169821679:+ [[0.8333333333333334], [0.125], [0.041666666666666664]]
			#	ENSG00000000457.13;SE:chr1:169854964-169855796:169855957-169859041:- [[0.625], [0.3541666666666667], [0.020833333333333332]]

		## Creating another dictionary with its brain regions as a key
		my_probabilities_dict.setdefault(key_primary_bregion, pmydict_littlefile)   ## Saving my primary keys (ENS events) and the ENS + probabilities
	## EXAMPLE :
	## Brain_-_Cerebellum {'ENSG00000000003.14;SE:chrX:100630866-100632485:100632568-100633405:-': [[0.9583333333333334], [0.020833333333333332], [0.020833333333333332]],...

		## Creating another dictionary with its brain regions as a key and the 188 line dictionary
		my_ntotal_dict.setdefault(key_primary_bregion, pmydict_wholefile)     ## Saving my primary keys (ENS events) and the ENS + total n
	#for key, value in my_ntotal_dict.items():
	## EXAMPLE :
	## Brain_-_Anterior_Cingulate_Cortex_(Ba24) {'ENSG00000000003.14;SE:chrX:100630866-100632485:100632568-100633405:-': [['up', 45], ['down', 0], ['NA', 0]], 'ENSG00000001084.10_and_ENSG00000231683.6;SE:chr6:53514497-53516109:53516222-53520778:-': [['up', 45], ['down', 0], ['NA', 0]],...

	# 2.7 Creating the list with the keys of the dictionaries got before
	region_ntotal=[my_ntotal_dict[regions[0]],my_ntotal_dict[regions[1]],my_ntotal_dict[regions[2]],my_ntotal_dict[regions[3]],my_ntotal_dict[regions[4]],my_ntotal_dict[regions[5]],my_ntotal_dict[regions[6]],my_ntotal_dict[regions[7]],my_ntotal_dict[regions[8]],my_ntotal_dict[regions[9]],my_ntotal_dict[regions[10]],my_ntotal_dict[regions[11]],my_ntotal_dict[regions[12]]]

	region_training=[my_probabilities_dict[regions[0]],my_probabilities_dict[regions[1]],my_probabilities_dict[regions[2]],my_probabilities_dict[regions[3]],my_probabilities_dict[regions[4]],my_probabilities_dict[regions[5]],my_probabilities_dict[regions[6]],my_probabilities_dict[regions[7]],my_probabilities_dict[regions[8]],my_probabilities_dict[regions[9]],my_probabilities_dict[regions[10]],my_probabilities_dict[regions[11]],my_probabilities_dict[regions[12]]]

    # 2.8 Creates the splicing event sets with every dictionary region keys
	ENS_events = set(my_ntotal_dict[regions[1]].keys()).intersection(set(my_ntotal_dict[regions[0]].keys()),set(my_ntotal_dict[regions[1]].keys()), set(my_ntotal_dict[regions[3]].keys()),set(my_ntotal_dict[regions[4]].keys()),set(my_ntotal_dict[regions[5]].keys()),set(my_ntotal_dict[regions[6]].keys()),set(my_ntotal_dict[regions[7]].keys()), set(my_ntotal_dict[regions[8]].keys()),set(my_ntotal_dict[regions[9]].keys()),set(my_ntotal_dict[regions[10]].keys()),set(my_ntotal_dict[regions[11]].keys()),set(my_ntotal_dict[regions[12]].keys())) ## In this case, you want to take repeated events in every Brain region and then you have to take the INTERSECTION, not the Union !!!

    # 2.9 Opening the dictionaries which contains the main values of the model
	first_ENS_dict={}
	p_conditional={}

    ## For-loop creates for get the common events
	for ENS_event in ENS_events:
		psi_up=0
		psi_down=0
		psi_na=0
		psi_main_list=[]

        # 2.9.1 Iterating region_ntotal
		for reg in region_ntotal:

            # 2.9.1.1 Checking the coincidences
			if ENS_event in reg:

                ## Localizing the position of up, downs and NA and adding the value
				psi_up += reg[ENS_event][0][1]
				psi_down += reg[ENS_event][1][1]
				psi_na += reg[ENS_event][2][1]

        # 2.9.2 Formula which computes the probability of each attribute. It has to add pseudocounts (Laplace rules) in order to avoid the division by zero in the denominator of Naive Bayes posterior probability formula.
		first_ENS_dict[ENS_event]=[["up",((psi_up+1)/((psi_up+psi_down+psi_na)+3))], ["down", ((psi_down+1)/((psi_up+psi_down+psi_na)+3))],["NA",(psi_na+1)/((psi_up+psi_down+psi_na)+3)]]

        # 2.9.3 Iterating region_ntraining
		for reg in region_training:

			# 2.9.3.1 Posterior probability formula -> p(regionbrain|SE)=(P(SE|regionbrain)*(P(regionbrain))/P(SE)). P(regionbrain) is the class prior probability (1/13)
			psi_main_list.append([["up",reg[ENS_event][0][0]/(first_ENS_dict[ENS_event][0][1]*13)],["down", reg[ENS_event][1][0]/(first_ENS_dict[ENS_event][1][1]*13)],["NA",reg[ENS_event][2][0]/(first_ENS_dict[ENS_event][2][1]*13)]])

        # 2.9.4 Finally it creates a dictionary with the posterior probability and each ENS event
		p_conditional.setdefault(ENS_event,psi_main_list)


        # 2.9.5 It starts to initialize the entropy variables
		entropy_region=mt.log2(len(regions))

		entropy_region_event={}
		MI_region_event={}

		for key_entropy, value_entropy in p_conditional.items():

			entropy_up=0
			entropy_down=0
			entropy_NA=0

            # 2.9.5.1 For value_entropy it computes the logarithm of each psi_main_list values multiplain by the same values and adding it in entropy up
			for reg in value_entropy:
				entropy_up+=reg[0][1]*mt.log2(reg[0][1])
				entropy_down+=reg[1][1]*mt.log2(reg[1][1])
				entropy_NA+=reg[2][1]*mt.log2(reg[2][1])

            # 2.9.5.2 Formula which computes the entropy for each event in its region
			entropy_region_event[key_entropy]=((first_ENS_dict[key_entropy][0][1]*(-1)*entropy_up) + (first_ENS_dict[key_entropy][1][1]*(-1)*entropy_down)+(first_ENS_dict[key_entropy][2][1]*(-1)*entropy_NA))

            # 2.9.5.2 Calculating mutual information
			MI_region_event[key_entropy]=entropy_region-entropy_region_event[key_entropy]

        # 2.9.6 Mapping the values into a list and delimiting a threshold for the mutual information values that will compute the score of each prediction after. For our model, the proposal value of the percentile is 25
		MUTUALINFORMATION_list=list(map(float, MI_region_event.values()))
		MUTUALINFORMATION_list=np.array(MUTUALINFORMATION_list)
		MUTUALINFORMATION_threshold=np.percentile(MUTUALINFORMATION_list,25)

        # 2.9.7 Iterating the last dictionary and saving each mutual information in a list, in order to discard the non-selected vues
		model_event=[]
		for event, MI_value in MI_region_event.items():

			if MI_value >=MUTUALINFORMATION_threshold:
				model_event.append(event)

    # 2.10 It select the most information content splicing events in order to apply them to the test samples
	print("I have chosen %s splicing events !!!!!!\nIf you don't want to continue executing the program, please click CTRL+Z" %(len(model_event)))

	y_actual_list=[]
	y_predict_list=[]
	my_final_results=[]

    ## Dictionary creates at the first lines of the script
	for xxx, values in RegionSamples_test.items():
		for sampletest in values:
			sampletest_prob=[]
			for region in region_training:
				region_prob=0.0

                ## Selecting the most information content events
				for event in model_event:

					if event in testing_dict[sampletest][0][1]:
						region_prob+=mt.log2(float(region[event][0][0])*(1/13))
					elif event in testing_dict[sampletest][1][1]:
						region_prob+=mt.log2(float(region[event][1][0])*(1/13))
					elif event in testing_dict[sampletest][2][1]:
						region_prob+=mt.log2(float(region[event][2][0])*(1/13))
				sampletest_prob.append(region_prob)     # Preparing it for create an score list

            ## My score prediction list
			my_score=max(sampletest_prob)
			score_index=sampletest_prob.index(my_score)
			my_prediction=regions[score_index]

            ## Adding the labels of my known file to compare with the predictions
			for region2 in RegionSamples_test:
				if sampletest in RegionSamples_test[region2]:
					my_label=region2

			## Creating the list of prediction and labels
			y_actual_list.append(my_label)
			y_predict_list.append(my_prediction)


			my_final_results.append((my_score, my_prediction, my_label, sampletest))
			my_final_results.sort(reverse=True) # Ordered the scores

    # 2.11 Printing the results in a csv file
	df = pd.DataFrame(my_final_results, columns=["score", "prediction" , "label", "sample"])
	df.to_csv("finalpredictions.csv")

    # 2.12 Finally, it creates and plots the confusion matrix and getting the accuracy results
	y_actual=pd.Series(y_actual_list, name ="Actual")
	y_predic=pd.Series(y_predict_list, name ="Prediction")
	df_confusion= pd.crosstab(y_actual, y_predic, rownames=["Actual"], colnames=["Prediction"], margins=True)
	df_conf_norm=df_confusion/df_confusion.sum(axis=1)
	plt.matshow(df_conf_norm, cmap="Purples") # imshow
	#plt.title("Accuracy plot - Naive Bayes - Brain")
	plt.colorbar()
	tick_marks = np.arange(len(df_conf_norm.columns))
	plt.xticks(tick_marks, df_conf_norm.columns, rotation=90)
	plt.yticks(tick_marks, df_conf_norm.index)
	#plt.tight_layout()
	plt.ylabel(df_conf_norm.index.name)
	plt.xlabel(df_conf_norm.columns.name)
	plt.show()
	#2.13 Prints Statistics for our Confusion Matrix
	cm = CM(y_actual, y_predic)
	cm.print_stats()

# 3. Calling the function. The proposed files have been writen.
Naive_Bayes_Brain("trainingtest01_29males_21females_650samples.txt","testingtest01_88samples.txt","gtex_brain_samples_formatted.psi")
