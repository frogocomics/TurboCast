# TurboCast: High-Speed Short-Term Weather Prediction with CNN-LSTM
__BANJO Analytics__: Jeff Chen, Alec Mak, Norbert Irisk, Matthew Pascal
___
_TurboCast_ is a high performance, short-term localized weather forecasting model using a hybrid CNN-LSTM architecture.
This model uses both weather radar and weather station input to predict SOMETHING. To deal with missing radar images, a
neural network _TurboImpute_ is trained.

![Project structure](./images/project_structure.svg)

A full detailed summary of the model can be found at LINK HERE.

![Architecture](./images/architecture.svg)

### File Structure
![File structure](./images/filetree.svg)

### Data Fetching
Although the processed weather station data is already available in the [src/data/weatherstation/processed](src/data/weatherstation/processed)
directory, users have the option of regenerating the data through the following commands. Due to very large file sizes, 
radar data is available from [this link](dab) and should be placed in
[src/data/weatherstation/processed](src/data/weatherstation/processed) for processed files and in
[src/data/weatherstation/download](src/data/weatherstation/download) for raw month-by-month folders.

For weather station web scraping:
```bat
python src/data/weatherstation/weather_scraper.py n 8 --start_month 2012-1 --end_month 2023-10
```

For weather station processing and imputation (single-threaded implementation):
```bat
python src/data/weatherstation/weather_process.py
```
Note that this will not work without the full time range of data!

For radar image web scraping:
```bat
python src/data/radar/radar_scraper.py -n 8 --start_month 2012-1 --end_month 2023-10 --weather rain
```
For radar image processing:
```bat
python src/data/radar/radar_processor.py -n 8 --start_month 2012-1 --end_month 2023-10
```
Where the command line parameters are:

| Argument                                | Format         | Description                         | Default |
|-----------------------------------------|----------------|-------------------------------------|---------|
| ```-n``` _or_ ```--n_threads```         | Integer        | Number of threads to use            | 1       |
| ```--start_month```                     | YYYY-MM        | First month to scrape/process       | N/A     |
 | ```--end_month```                       | YYYY-MM        | Last month to scrape/process (inclusive) | N/A     |
| ```--weather``` (Only for radar) | rain _or_ snow | Weather type to scrape              | rain    |

> Note that radar web scraping must be run twice before image processing to download both rain and snow data!

### Model Training
Model training code is detailed in ```train_model.ipynb``` in the [model](src/model) folder and details the model
structure and data input pipeline. However, trained model weights are already provided.

### Model Evaluation
Code for running model evaluation is provided in ```eval_model.ipynb``` in the [model](src/model) folder and provides
a basic structure for evaluating on new data.

### Dependencies
Lorum ipsum

