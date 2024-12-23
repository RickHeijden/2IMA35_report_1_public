import os, subprocess
from typing import TextIO

PYTHON_FILE = 'MSTforDenseGraphs/PysparkMSTfordensegraphsfast.py'

def run(args):
    result:str = subprocess.run(
        ['.venv/Scripts/python.exe', PYTHON_FILE] + args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True
    ).stderr
    result: list[str] = result.split('\n')
    result: list[str] = list(filter(lambda x: x != '' and ';' in x and 'WARN' not in x, result))
    return result

def run_experiment(reduce_edge_function: str, names_datasets: list[str], num_clusters: list[str]) -> list[str]:
    args = []
    args.append('--reduce_edge_function')
    args.append(reduce_edge_function)
    args.append('--names_datasets')
    args.extend(names_datasets)
    args.append('--num_clusters')
    args.extend(num_clusters)

    result = run(args)
    print(result)

    return result

def main():
    possible_reduce_edge_functions = [
        'PURE_PYTHON',
        'FULL_SPARK',
        'INTERMEDIATE_COLLECT',
        'CONFIG',
        'DATAFRAMES',
        'SINGLE_PARTITION',
        'SINGLE_FUNCTION',
        'SINGLE_SLICE',
    ]

    names_datasets = ['2d-20c-no0', '2sp2glob', '3-spiral', 'D31', 'spiralsquare', 'square1', 'twenty', 'fourty']
    num_clusters = ['20', '4', '3', '31', '6', '4', '20', '40']

    directory_name = 'experiments'
    if os.path.exists(f"results/{directory_name}.csv"):
        os.remove(f"results/{directory_name}.csv")
    with open(f"results/{directory_name}.csv", "w") as f:
        unbuffered_write(f, "function,dataset,num_clusters,mst_time,total_time,vertices,edges\n")

        for reduce_edge_function in possible_reduce_edge_functions:
            for i in range(len(names_datasets)):
                results = run_experiment(reduce_edge_function, [names_datasets[i]], [num_clusters[i]])
                for result in results:
                    size_and_timings = result.split(';')
                    dataset = size_and_timings[0]
                    size = size_and_timings[1]
                    edge_size = size_and_timings[2]
                    mst_time = size_and_timings[3]
                    total_time = size_and_timings[4]
                    dataset_num_clusters = num_clusters[names_datasets.index(dataset)]
                    unbuffered_write(
                        f,
                        f"{reduce_edge_function},{dataset},{dataset_num_clusters},{mst_time},{total_time},{size},{edge_size}\n"
                    )


def unbuffered_write(file: TextIO, string: str):
    file.write(string)
    file.flush()
    os.fsync(file.fileno())


if __name__ == '__main__':
    main()
