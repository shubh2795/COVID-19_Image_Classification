I've completed the project in a way such that the project_01 functionalites are intact.

Here is a list of steps to run the project:(Step 1- Step 3 are optional) 
1. Download the dataset from kaggle:
    https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset
2. Open data_conversion.py and update the path variables respectively.
3. Run data_conversion.py : This gives the data in the form of pck files so that we can use this datasets for getting our models in a similar way as we did for project01.

Alternatively you can skip step 1:3 as I am uploading the pck files for training, testing and validation on box.
Box link for pck files: https://usu.box.com/s/fsk9glwbjpbftfw36hzh43ozpy08649i

4: Use the pck files 

5. I have relative paths set throughout the project, in case you get errors,update the base_path variable as per your file structure. 

6. First lets apply random forest technique, The code for random forests is in “random_forests_and_decision_trees.py” and “covid_data_loader_rf.py” contains the code for loading the dataset as we did in mnist_loader.py in hw07 and this code can be run through the “covid_rf_uts.py” 
Run: "python3.7 covid_rf_uts.py" You can also run covid_rf_uts.py through pycharm.

7. Let's move to getting the required CNN and ANN models:
 	a) (Optional) Go to script.py and edit the epochs and batch size in the  get_ann_models() and get_cnn_models().
	b) (Optional) Change the layers and other parameter like learning rate in project2_image_anns.py or project2_image_cnns.py 		   respectively.
	c) uncomment line 23 or line 24 (#get_ann_models(), #get_cnn_models() whatever you want to run)
	d) run python3.7 script.py to train the model and obtain validation accuracy. or run scrpt.py directly through pycharm.


8. I have added my best models and their accuracy summary in load_nets.py.
9. test_load_nets.py can be used to test the load_nets.py and obtain accuracies for all the models.
Just uncomment one load_ann_model() or load_cnn_model() at a time in test_load_nets.py and then run python3.7 test_load_nets.py to get the accuracy of a particular model.
If you run all of them together it throws an error to load 6 models at one time.

10. covid_ensemble.py has ensembles for ANN's, CNN's respectively, ANN's + CNN's + Decision Trees. Just run using "python3.7 covid_ensemble.py" or covid_ensemble.py run directly through pycharm.




