# Sargasses dataset

The sargasses dataset is made of OTCI images annotated by mfai experts.
It has not beed made public yet.

## The data.
The dataset is composed of an OTCI image and sargasum mask pair.

![](images/data_example.png)

OTCI (OLCI Terrestrial Chlorophyll Index) images represent the chlorophyll density at the surface of the earth. It is used to detect sargasum as it is an algae that contain chlorophyll. These images come from the OLCI measuring tool on the sentinel 3 satelite.

The sargasum masks were made by mfai experts and are binary masks that shows where on the OTCI image sargasum is present or not.

You can find an exemple OTCI image at [tests/data/20220401S3_OTCI.png](../tests/data/20220401S3_OTCI.png) and an exemple sargasum mask at [tests/data/20220401_w1.npz](../tests/data/20220401_w1.npz).


## Loading the data
### Sample class [[source](../sargasses/sample.py)]
The sample class is responsible for loading data from disk. It loads and holds one otci image and its corresponding target mask for one date.

It is used by the dataset class.



### Dataset class [[source](../sargasses/dataset.py)]
A pytorch dataset implementation. Is responsible for iterating over the data of either training, validation or test.

It is used by the datamodule class.


### Datamodule class[[source](../sargasses/datamodule.py)]
A pytorch lightning datamodule implementation. Is responsible for instantiationg the datasets and wrapping them in a pytorch dataloader.

It is instantiated by the lightning cli.