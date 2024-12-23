# 2IMA35_reports

## Reduce edge functions
All the reduce edge function detailed in our report can be found in `MSTforDenseGraphs/reduceEdges.py`.

We have adjusted the code in `PysparkMSTfordensegraphsfast.py` such that it can use these functions.

## Running the code
The requirements of this code is `python3.11` and the other packages are listed in `requirements.txt`.

To run our experiments, you can run the `experiment.py` script.
The results of the experiments can be found in the `results` folder. The results are stored in `.csv` files.

Do note that you need to create a virtual environment with all the requirements installed. This is to ensure that 
the `experiment.py` can find the `.venv/Scripts/python.exe` file. Also, Spark is required to be installed on your machine.

## Visualizations
To visualize the results, you can run the `visualize_results.py` script.

The results of these visualizations can be found in the `visualizations` folder.
