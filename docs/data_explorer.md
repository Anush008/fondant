# Data explorer

## Data explorer UI

The data explorer UI enables Fondant users to explore the inputs and outputs of their Fondant pipeline.

The user can specify a pipeline and a specific pipeline run and component to explore. The user will then be able to explore the different subsets produced by by Fondant components.

The chosen subset (and the columns within the subset) can be explored in 3 tabs.

![data explorer](../art/data_explorer/data_explorer.png)

## How to use?
You can setup the data explorer container with the `fondant explore` CLI command, which is installed together with the Fondant python package.

=== "Console"

    ```bash
    fondant explore --base_path $BASE_PATH
    ```

=== "Python"

    ```python
    from fondant.explore import run_explorer_app
    
    BASE_PATH = "your_base_path"
    run_explorer_app(base_path=BASE_PATH)
    ```

Where the base path can be either a local or remote base path. Make sure to pass the proper mount credentials arguments when using a remote base path or a local base path 
that references remote datasets. You can do that either with `--auth-gcp`, `--auth-aws` or `--auth-azure` to
mount your default local cloud credentials to the pipeline. Or You can also use the `--credentials` argument to mount custom credentials to the local container pipeline.

Example: 

=== "Console"

    ```bash
    export BASE_PATH=gs://foo/bar
    fondant explore --base_path $BASE_PATH
    ```

=== "Python"

    ```python
    from fondant.explore import run_explorer_app
    
    BASE_PATH = "gs://foo/bar"
    run_explorer_app(base_path=BASE_PATH)
    ```

### Sidebar
In the sidebar, the user can specify the path to a manifest file. This will load the available subsets into a dropdown, from which the user can select one of the subsets. Finally, the columns within the subset are shown in a multiselect box, and can be used to remove / select the columns that are loaded into the exploration tabs.

### Data explorer Tab
The data explorer shows an interactive table of the loaded subset DataFrame with on each row a sample. The table can be used to browse through a partition of the data, to visualize images inside image columns and more.

### Numeric analysis Tab
The numerical analysis tab shows statistics of the numerical columns of the loaded subset (mean, std, percentiles, ...) in a table. In the second part of the tab, the user can choose one of the numerical columns for in depth exploration of the data by visualizing it in a variety of interactive plots.

![data explorer](../art/data_explorer/data_explorer_numeric_analysis.png)

### Image explorer Tab
The image explorer tab enables the user to choose one of the image columns and analyse these images.

![data explorer](../art/data_explorer/image_explorer.png)