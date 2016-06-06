# seismos for NFP Stability
####_Paul Mack, Sam Sun, Alden Golab_
A prediction system for anticipating not-for-profit service provider fiscal stability year-over-year. 
Final project for Machine Learning in Public Policy, Spring 2016.

## Requirements

We have included a `requirements.txt` file in the setup folder, which contains pip statements that ought to be run prior to doing any work with the files contained in this repo. All code has been written in Python 3.

## Getting the Data

Data are provided by:

US Internal Revenue Service (IRS): 
+ [IRS 990 Extracts Calendar Year 2012-2015](https://www.irs.gov/uac/soi-tax-stats-annual-extract-of-tax-exempt-organization-financial-data)
+ [IRS Exempt Organizations Business Master File for Calendar Year 2016 - One File Per Region](https://www.irs.gov/Charities-&-Non-Profits/Exempt-Organizations-Business-Master-File-Extract-EO-BMF)

US Department of Commerce:
+ [City GDP Per Capita](http://www.bea.gov/regional/)

By running `merge_years.py` with the data from the setup folder, you should be able to replicate the exact raw data we utilized for this project. The following files are explicitly called by `merge_years.py`:

+ py12_990.dat
+ py13_990.dat
+ py14_990.dat
+ 15eofinextract990.dat
+ eo1.csv
+ eo2.csv
+ eo3.csv
+ eo4.csv

Not included in this are the GDP per Capita: these were separately merged into the zipmsa.csv file and renamed. We will add support for that eventually; however, it's a relatively easy process to carry out on your own if you need to. 

Strangely, and perhaps worrisomely, the IRS data contains repeated EINs; that means that organizations appear twice in the data, oftentimes with competing values. We de-duplicate on EINs as these are the de-facto unique IDs for the data and keep the first entry for the EIN we encounter. 

## Building Features

Also contained with the setup folder is the `feature_generation.py` file. This is a fully automated feature generation for the project, using the product of `merge_years.py` to create a new .csv with solely the features to be used. 

## Modeling

The model folder contains the code used to run our models. `acg_process.py` contains a model loop, requiring a features set input and will print out to screen results from selected models. As it is set, it will run all models for all features and carry out the necessary data transformations and cleaning. 

From this list, we then run `model_pickle.py` which takes specifications for the model that performed best in `acg_process.py` and re-runs against splits of the dataset in a Monte-Carlo style simulation. We modified this file on the fly for running Monte-Carlo cross-validation with a pickled file -- we will work to add support for this functionality, but it is not here yet. 

`run_validation.py` runs the pickeled model against a validation year. This is imperative: given the structure of IRS data reporting, a model must be able to predict on the following year using a model trained on previous years: the data for the following year won't be available until __the year after that__. 
