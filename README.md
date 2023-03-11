# Inventory Monitoring at Distribution Centers - Counting Objects in Amazon Bin Image Dataset using AWS SageMaker

This project aims to demonstrate how to use Amazon SageMaker to develop and deploy a machine learning model for counting the number of objects in images of bins from the Amazon Bin Image Dataset. The focus of the project is on the use of Amazon SageMaker to perform various tasks such as dataset preparation, hyperparameter tuning, model training, model evaluation, and model deployment.

![ex1](https://github.com/silverbottlep/abid_challenge/blob/master/figs/image1_small.jpg?raw=true)

**Note:** *Please note that the goal of this project is to demonstrate the use of Amazon SageMaker, and not to create a highly accurate machine learning model. As such, the results presented in this project may not be representative of the best possible performance that could be achieved with the dataset and the chosen model*.

## File Structure

- `bin-object-counting.ipynb`: Jupyter notebook that demonstrates how to use Amazon SageMaker to preprocess the dataset, train the model, and deploy the endpoint.

- `process_data.py`: Python script that is used as the entry point for an Amazon SageMaker Processing Job. This script extracts a subset of the Amazon Bin Image Dataset with desired labels 0-5 and saves the output to `val.json`.

- `hpo.py`: Python script that is used as the entry point for an Amazon SageMaker Hyperparameter Tuning Job. This script defines the hyperparameter search job configuration.

- `train_model.py`: Python script that defines the model architecture, training loop, and evaluation metrics. This script is used to train the ResNet50 model using PyTorch and Amazon SageMaker PyTorch training.

- `inference.py`: Python script that defines the endpoint prediction function. This script is used to generate predictions from the deployed endpoint.

- `ProfilerReport/`: Directory that contains the profiler report generated by Amazon SageMaker Profiler.

- `inference_output.zip`: ZIP file that contains the test set inference results generated by Amazon SageMaker Batch Transform.

- `figs/`: Directory that contains supporting images.

- `val.json`: JSON file that contains a list of image file names and their corresponding labels from the subset of the Amazon Bin Image Dataset.

- `requirements.txt`: File that lists the Python libraries required to run the project.

## Project Set Up and Installation

To set up this project, you need to have access to Amazon SageMaker. You can create a SageMaker notebook instance or use the SageMaker Studio. If you prefer to run the project locally, you will need to install the AWS CLI and configure it to use SageMaker. Here are the steps to do that:

1. Install the AWS CLI by following the instructions in the AWS CLI User Guide.

2. Once the CLI is installed, open a terminal or command prompt and run aws configure to set up your credentials. You will need to provide your Access Key ID, Secret Access Key, default region name, and default output format. You can find your Access Key ID and Secret Access Key in the IAM console.

3. Next, you can clone this repository and navigate to the project directory:

    ```
    git clone https://github.com/your-username/your-repository.git
    cd your-repository
    ```

4. Create a virtual environment and activate it:

    ```
    python3 -m venv env
    source env/bin/activate
    ```

5. Install the required packages:

    ```pip install -r requirements.txt```

6. To run the Jupyter notebook locally, run the following command:

    `jupyter notebook`

This will open a new browser window with the Jupyter notebook interface. You can then navigate to the `bin-object-counting.ipynb` notebook and run it.

Note that if you plan to use Amazon SageMaker for this project, you don't need to install the AWS CLI or configure it. Instead, you can follow the instructions in the notebook to create a SageMaker notebook instance and run the code directly in the notebook.

## Dataset

### Overview

The Amazon Bin Image Dataset contains images of bins containing objects of various types and sizes. The dataset is available for public use and can be accessed from the Amazon S3 bucket [here](https://registry.opendata.aws/amazon-bin-imagery/).

### Access

To access the dataset, you can use the boto3 library in Python to interact with the S3 bucket. In this project, the **SageMaker Processing Job** is used to perform ETL (Extract, Transform, and Load) on the dataset.

### Subset

Since the Amazon Bin Image Dataset is quite large, this project is using a subset of the data to conserve AWS costs. Specifically, the project is using 10% of the original dataset.

### Classes

In this project, the focus is on counting objects in bins, so the classes being used range from 0-5. These classes represent the number of objects in each bin.

![distribution](figs/class_distribution.png)

## Model Training

For this project, transfer learning was used to improve the accuracy of the model. Specifically, the ResNet50 model was used as the base model and fine-tuned on the Amazon Bin Image Dataset. The model was implemented using PyTorch and trained using Amazon SageMaker PyTorch training.

### Hyperparameters

To optimize the performance of the model, a hyperparameter tuning job was launched using Amazon SageMaker Hyperparameters Tuner. The following hyperparameters were tuned:

- Learning Rate (`lr`): the step size used during training
- Batch Size (`batch_size`): the number of images used in each training iteration

After tuning, the best hyperparameters were found to be `lr = 0.0033` and `batch_size = 128`.

In addition, the model was trained for 50 epochs to balance accuracy and cost.

### Class Imbalance

One challenge encountered in this project was the class imbalance present in the dataset. To address this issue, several techniques were used. First, a weighted sampler was used during training to ensure that each class was represented equally. Second, the scheduler and optimizer were adjusted to account for class imbalance. Third, a criterion weight tensor was used to weight the loss function and give more weight to underrepresented classes.

Additional details about these techniques can be found in the `train_model.py` file. Overall, these methods helped to address the class imbalance and improve the accuracy of the model.

### Evaluation

To evaluate the performance of the model, Amazon SageMaker Batch Transform was used to run the model on a set of test data. The model achieved an accuracy of 31% and a root mean square error (RMSE) of 1.96.

The per-class accuracy and RMSE were also calculated:

| Quantity | Per class accuracy(%) | Per class RMSE |
|----------|----------------------|----------------|
| 0        | 74.74                | 1.08           |
| 1        | 28.66                | 0.99           |
| 2        | 70.82                | 1.47           |
| 3        | 23.18                | 1.53           |
| 4        | 20.12                | 1.85           |
| 5        | 19.08                | 2.13           |

These results provide insights into how well the model performs for each class, which can be useful for understanding where the model struggles and how to improve its accuracy.

## Machine Learning Pipeline

The machine learning pipeline for this project consists of the following steps:

- Data preparation: A subset of the Amazon Bin Image Dataset was created and preprocessed using Amazon SageMaker Processing Job.

- Hyperparameter tuning: Amazon SageMaker Hyperparameters Tuner was used to find the best hyperparameters for the model, including the learning rate and batch size.

- Model training: The ResNet50 model was used as the base model and fine-tuned using PyTorch and Amazon SageMaker PyTorch training. The model was trained for 50 epochs.

- Class imbalance: Several methods were used to address the class imbalance present in the dataset, including a weighted sampler, a criterion weight tensor, and adjustments to the optimizer and scheduler.

- Model evaluation: The performance of the model was evaluated using Amazon SageMaker Batch Transform. The model achieved an accuracy of 31% and a root mean square error (RMSE) of 1.96. Per-class accuracy and RMSE were also calculated to provide insights into how well the model performs for each class.

- Model deployment: The model was deployed using Amazon SageMaker Endpoint.

Overall, this pipeline provides a framework for developing and deploying machine learning models for various computer vision tasks. The specific details of each step may vary depending on the task and the dataset, but the overall process remains the same.

## Standout Suggestions

In addition to the main pipeline, several standout suggestions were implemented in this project:

1. Multi-instance training: Amazon SageMaker multi-instance training was used to speed up the training process by distributing the training workload across multiple instances. This can be a useful technique when training large models or datasets.

2. Deploying the model endpoint: In the notebook, detailed instructions were provided on how to invoke the deployed endpoint, including how to prepare and send requests and how to handle responses.

3. Profiler and debugger: Amazon SageMaker Profiler and Debugger were used to monitor the training process and identify performance bottlenecks and bugs. These tools can be useful for improving the efficiency and accuracy of the model.

Overall, these standout suggestions demonstrate additional techniques and tools that can be used to further improve the machine learning pipeline and the performance of the model.
