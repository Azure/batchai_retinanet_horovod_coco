# Distributed training of a Fizyr Keras-RetinaNet model on the COCO dataset using Horovod on Azure Batch AI

## Overview

This repository demonstrates how to train a RetinaNet object detection model using [Horovod](https://github.com/uber/horovod) on the [Azure Batch AI](https://azure.microsoft.com/services/batch-ai/) platform. We use [Fizyr](https://fizyr.com/)'s [Keras-RetinaNet](https://github.com/fizyr/keras-retinanet/tree/master/keras_retinanet) implementation of RetinaNet for the model architecture and training method, with minor modifications to allow distributed training via Horovod. For more information and motivation on our approach, please see our [blog post](https://blogs.technet.microsoft.com/machinelearning/2018/06/20/how-to-do-distributed-deep-learning-for-object-detection-using-horovod-on-azure/).

## Set up the Azure Batch AI cluster

### Prerequisites

**Azure Subscription**

This tutorial will require an [Azure subscription](https://azure.microsoft.com/en-us/free/) with sufficient quota to create a storage account and two NC24 VMs as a Batch AI cluster.

**Files from this repository**

You will need local copies of the .json and .py files included in this GitHub repository. We recommend that you download or clone the full repository locally, but you can also download each file individually. (If you choose that approach, be careful to download the "raw" files -- it's common to accidentally save GitHub's HTML previews of the files instead.)

**Utilities**

This tutorial requires the following programs:
- [Azure CLI 2.0](https://docs.microsoft.com/cli/azure/install-azure-cli)
- [AzCopy](https://docs.microsoft.com/azure/storage/common/storage-use-azcopy)

These programs are available for Windows and Linux. If you prefer not to install these programs locally, you may instead provision an [Azure Windows Data Science Virtual Machine](https://docs.microsoft.com/azure/machine-learning/data-science-virtual-machine/provision-vm). (Both programs are pre-installed on this VM type and are available on the system path.) The commands included in this tutorial were written and tested in Windows, but readers will likely find it straightforward to adapt for Linux.

Once these programs are installed, open a command line interface and check that the binaries are available on the system path by issuing the commands below:
```
az
azcopy
```
If not, you may need to [edit the system path](http://www.zdnet.com/article/windows-10-tip-point-and-click-to-edit-the-system-path-variable/) to point to the folders containing these binaries (e.g., `C:\Program Files (x86)\Microsoft SDKs\Azure\AzCopy`) and load a fresh command prompt.

### Prepare to use the Azure CLI

In your command line interface, execute the following command. The output will contain a URL and token that you must visit to authenticate your login.
```
az login
```

You will now indicate which Azure subscription should be charged for the resources you create in this tutorial. List all Azure subscriptions associated with your account:
```
az account list
```

Identify the subscription of interest in the JSON-formatted output. Use its "id" value to replace the bracketed expression in the command below, then issue the command to set the current subscription.
```
az account set -s [subscription id]
```

Register the Batch/BatchAI providers and grant Batch AI "Network Contributor" access on your subscription using the following commands. Note that you will need to copy your subscription's id in place of the bracketed expression before executing the command.
```
az provider register -n Microsoft.Batch
az provider register -n Microsoft.BatchAI
az role assignment create --scope /subscriptions/[subscription id] --role "Network Contributor" --assignee 9fcb3732-5f52-4135-8c08-9d4bbaf203ea
```

It may take ~10 minutes for the provider registration process to complete. You may proceed with the tutorial in the meantime.

### Create an Azure resource group

We will create all resources for this tutorial in a single resource group, so that you may easily delete them when finished. Choose a name for your resource group and insert it in place of the bracketed expression below, then issue the commands:
```
set AZURE_RESOURCE_GROUP=[resource group name]
az group create --name %AZURE_RESOURCE_GROUP% --location eastus
```
You may use other locations, but we recommend `eastus` for proximity to the data that will be copied into your storage account, and because the necessary VM type (NC series) is available in the East US region.

### Create an Azure storage account and populate it with files

We will create an Azure storage account to hold training and evaluation data, scripts, and output files. Choose a unique name for this storage account and insert it in place of the bracketed expression below. Then, issue the following commands to create your storage account and store its randomly-assigned access key:
```
set STORAGE_ACCOUNT_NAME=[storage account name]
az storage account create --name %STORAGE_ACCOUNT_NAME% --sku Standard_LRS --resource-group %AZURE_RESOURCE_GROUP% --location eastus
for /f "delims=" %a in ('az storage account keys list --account-name %STORAGE_ACCOUNT_NAME% --resource-group %AZURE_RESOURCE_GROUP% --query "[0].value"') do @set STORAGE_ACCOUNT_KEY=%a
```

With the commands below, we will create an Azure File Share to hold scripts and logs, as well as an Azure Blob container for fast I/O during model training. (The file share offers more options for retrieving your log files, while data access will be faster from blob containers.)
```
az storage share create --account-name %STORAGE_ACCOUNT_NAME% --name batchaicoco
az storage container create --account-name %STORAGE_ACCOUNT_NAME% --name coco
```

Next, you'll use your favorite file transfer method to copy the necessary files to your storage account. (We recommend the Azure Portal or Azure Storage Explorer.) Create a folder named "scripts" in the "batchaicoco" file share, and upload the `train.py` file in the "scripts" folder. Upload the 2017 COCO dataset into the "coco" blob container with the following directory structure:
```
annotations/instances_train2017.json
annotations/instances_val2017.json
images/train2017/*.jpg
images/val2017/*/jpg
```

### Create the Batch AI workspace and experiment

The workspace will house your cluster and training jobs. Select a name for your workspace and execute the following command in the CLI:

```
set WORKSPACE_NAME=[your selected workspace name]
az batchai workspace create -n %WORKSPACE_NAME% --resource-group %AZURE_RESOURCE_GROUP% 
```

Your jobs will be further grouped under an experiment. Select a name for your experiment and execute the following command in the CLI:

```
set EXPERIMENT_NAME=[your selected experiment name]
az batchai experiment create -n %EXPERIMENT_NAME% -w %WORKSPACE_NAME% --resource-group %AZURE_RESOURCE_GROUP% 
```

### Create the Batch AI cluster

Modify the file `cluster.json` to include your storage account name and storage account key where indicated. (Your storage account key was saved in the local variable %STORAGE_ACCOUNT_KEY% by an earlier command.) Then, select a name for your cluster and execute the following command in the CLI:

```
set CLUSTER_NAME=[your selected cluster name]
az batchai cluster create -n %CLUSTER_NAME% --image UbuntuDSVM --resource-group %AZURE_RESOURCE_GROUP% -w %WORKSPACE_NAME%
```

Your cluster may take roughly ten minutes to provision. To check on progress, execute the command below to check whether both VMs have entered the "idle" state or are still being prepared:

```
az batchai cluster show -n %CLUSTER_NAME% -g  %AZURE_RESOURCE_GROUP% -w %WORKSPACE_NAME%
```

## Submit training jobs

If desired, modify the `training_job.json` file to use a specific number of GPUs and VMs. Then, select a name for your training job and execute the command below:

```
set JOB_NAME=[your selected job name]
az batchai job create -n %JOB_NAME% -c %CLUSTER_NAME% -g  %AZURE_RESOURCE_GROUP% -w %WORKSPACE_NAME% -e %EXPERIMENT_NAME% -f training_job.json
```

You can check that your job is running successfully using the command below:
```
az batch job show -n %JOB_NAME% -g %AZURE_RESOURCE_GROUP% -w %WORKSPACE_NAME% -e %EXPERIMENT_NAME%
```

You can also monitor the streaming output for your job with the commands below:
```
az batchai job file stream -d stdouterr -j %JOB_NAME% -n stdout.txt -g %AZURE_RESOURCE_GROUP% -w %WORKSPACE_NAME% -e %EXPERIMENT_NAME%
az batchai job file stream -d stdouterr -j %JOB_NAME% -n stderr.txt -g %AZURE_RESOURCE_GROUP% -w %WORKSPACE_NAME% -e %EXPERIMENT_NAME%
```

Finally, you can find the saved model checkpoints created by your job using the following command. The timestamps on the checkpoints can be used to find the epoch length:
```
az batchai job file list -j  %JOB_NAME% -d outputfiles -w %WORKSPACE_NAME% -e %EXPERIMENT_NAME%
```

The files can be downloaded using the URLs provided in this command's output, or using your favorite Azure file transfer utility such as Azure Portal or Azure Storage Explorer.

### Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
