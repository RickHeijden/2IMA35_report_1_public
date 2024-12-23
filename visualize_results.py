# Disclaimer this file was a work together with ChatGPT

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

result_directory = 'results/'
plot_directory = 'plots/'

# Load the CSV files
device1_data = pd.read_csv(f'{result_directory}experiments_device_1.csv')
device2_data = pd.read_csv(f'{result_directory}experiments_device_2.csv')

# Add the 'device' column
device1_data['device'] = 'device1'
device2_data['device'] = 'device2'


# Convert time columns to numeric (in seconds)
def convert_time_to_seconds(time_str):
    try:
        h, m, s = map(float, time_str.split(':'))
        return h * 3600 + m * 60 + s
    except:
        return None


device1_data['mst_time'] = device1_data['mst_time'].apply(convert_time_to_seconds)
device1_data['total_time'] = device1_data['total_time'].apply(convert_time_to_seconds)
device2_data['mst_time'] = device2_data['mst_time'].apply(convert_time_to_seconds)
device2_data['total_time'] = device2_data['total_time'].apply(convert_time_to_seconds)

# Combine the dataframes
combined_data = pd.concat([device1_data, device2_data], ignore_index=True)

# maximums for the axis
max_total_time = combined_data['total_time'].max() * 1.05
max_edges = combined_data['edges'].max() * 1.05
max_num_clusters = combined_data['num_clusters'].max() * 1.05
max_vertices = combined_data['vertices'].max() * 1.05

# Drop rows with invalid time data
combined_data = combined_data.dropna(subset=['mst_time', 'total_time'])

# Calculate the average metrics across devices, excluding non-numeric columns
time_columns = ['mst_time', 'total_time']
average_data = combined_data.groupby(['function', 'dataset', 'num_clusters', 'vertices', 'edges'])[time_columns].mean().reset_index()
average_data_device = combined_data.groupby(['function', 'dataset', 'num_clusters', 'vertices', 'edges', 'device'])[
    time_columns].mean().reset_index()

# Visualization 1: Improved Box plot for `mst_time` across functions
average_data = average_data.sort_values(by='mst_time')
fig1 = px.box(
    average_data,
    x='function',
    y='mst_time',
    title='MST Time Comparison by Function',
    labels={'mst_time': 'MST Time (s)', 'function': 'Function'},
    color='function',
)
# fig1.update_traces(boxmean='sd')  # Show mean and standard deviation in the box plot
fig1.update_layout(yaxis_title='MST Time (s)', xaxis_title='Function')
fig1.update_yaxes(range=[0, max_total_time])
fig1.write_image(f"{plot_directory}mst_time_boxplot.png")
fig1.show()

# Visualization 2: Improved Box plot for `total_time` across functions
average_data = average_data.sort_values(by='total_time')
fig2 = px.box(
    average_data,
    x='function',
    y='total_time',
    title='Total Time Comparison by Function',
    labels={'total_time': 'Total Time (s)', 'function': 'Function'},
    color='function',
)
# fig2.update_traces(boxmean='sd')  # Show mean and standard deviation in the box plot
fig2.update_layout(yaxis_title='Total Time (s)', xaxis_title='Function')
fig2.update_yaxes(range=[0, max_total_time])
fig2.write_image(f"{plot_directory}total_time_boxplot.png")
fig2.show()

# # Visualization 3: Scatter plot for `vertices` vs `mst_time` grouped by function
# average_data = average_data.sort_values(by='vertices')
# fig3 = px.scatter(
#     average_data,
#     x='vertices',
#     y='mst_time',
#     color='function',
#     title='Vertices vs MST Time by Function',
#     labels={'vertices': 'Number of Vertices', 'mst_time': 'MST Time (s)'}
# )
# fig3.update_xaxes(range=[0, max_vertices])
# fig3.update_yaxes(range=[0, average_data['total_time'].max() * 1.05])
# fig3.write_image(f"{plot_directory}mst_time_vertices_scatter_plot.png")
# fig3.show()
#
#
# # Visualization 4: Scatter plot for `edges` vs `total_time` grouped by function
# average_data = average_data.sort_values(by='edges')
# fig4 = px.scatter(
#     average_data,
#     x='edges',
#     y='total_time',
#     color='function',
#     title='Edges vs Total Time by Function',
#     labels={'edges': 'Number of Edges', 'total_time': 'Total Time (s)'}
# )
# fig4.update_xaxes(range=[0, max_edges])
# fig4.update_yaxes(range=[0, average_data['total_time'].max() * 1.05])
# fig4.write_image(f"{plot_directory}total_time_vertices_scatter_plot.png")
# fig4.show()

# Visualization 5: Line plot for `vertices` vs `mst_time` grouped by function
average_data = average_data.sort_values(by='vertices')
fig5 = px.line(
    average_data,
    x='vertices',
    y='mst_time',
    markers=True,
    color='function',
    title='Vertices vs MST Time by Function',
    labels={'vertices': 'Number of Vertices', 'mst_time': 'MST Time (s)'}
)
fig5.update_xaxes(range=[0, max_vertices])
fig5.update_yaxes(range=[0, max_total_time])
fig5.write_image(f"{plot_directory}vertices_vs_mst_time.png")
fig5.show()

# Visualization 5: Line plot for `vertices` vs `total_time` grouped by function
average_data = average_data.sort_values(by='vertices')
fig5_1 = px.line(
    average_data,
    x='vertices',
    y='total_time',
    markers=True,
    color='function',
    title='Vertices vs Total Time by Function',
    labels={'vertices': 'Number of Vertices', 'mst_time': 'MST Time (s)'}
)
fig5_1.update_xaxes(range=[0, max_vertices])
fig5_1.update_yaxes(range=[0, max_total_time])
fig5_1.write_image(f"{plot_directory}vertices_vs_total_time.png")
fig5_1.show()

# Visualization 6: Line plot for `edges` vs `total_time` grouped by function
average_data = average_data.sort_values(by='edges')
fig6 = px.line(
    average_data,
    x='edges',
    y='total_time',
    markers=True,
    color='function',
    title='Edges vs Total Time by Function',
    labels={'edges': 'Number of Edges', 'total_time': 'Total Time (s)'}
)
fig6.update_xaxes(range=[0, max_edges])
fig6.update_yaxes(range=[0, max_total_time])
fig6.write_image(f"{plot_directory}edges_vs_total_time.png")
fig6.show()

# Visualization 6.1: Line plot for `edges` vs `mst_time` grouped by function
average_data = average_data.sort_values(by='edges')
fig6_1 = px.line(
    average_data,
    x='edges',
    y='mst_time',
    markers=True,
    color='function',
    title='Edges vs MST Time by Function',
    labels={'edges': 'Number of Edges', 'mst_time': 'MST Time (s)'}
)
fig6_1.update_xaxes(range=[0, max_edges])
fig6_1.update_yaxes(range=[0, max_total_time])
fig6_1.write_image(f"{plot_directory}edges_vs_mst_time.png")
fig6_1.show()

# Visualization 6.2: Line plot for `edges` vs `total_time` grouped by function and device
average_data_device_function = combined_data.groupby(['edges', 'device', 'function'])[time_columns].mean().reset_index()
average_data_device_function = average_data_device_function.sort_values(by='edges')
fig6_2 = px.line(
    average_data_device_function,
    x='edges',
    y='total_time',
    markers=True,
    color='function',
    line_dash='device',
    title='Edges vs Total Time by Function and Device',
    labels={'edges': 'Number of Edges', 'total_time': 'Total Time (s)', 'function': 'Function', 'device': 'Device'}
)
fig6_2.update_xaxes(range=[0, max_edges])
fig6_2.update_yaxes(range=[0, max_total_time])
fig6_2.write_image(f"{plot_directory}edges_vs_total_time_function_device.png")
fig6_2.show()

# Visualization 6.3: Line plot for `edges` vs `mst_time` grouped by function for device 1
fig6_3_data = device1_data.groupby(['function', 'dataset', 'num_clusters', 'vertices', 'edges'])[time_columns].mean().reset_index()
fig6_3_data = fig6_3_data.sort_values(by='edges')
fig6_3 = px.line(
    fig6_3_data,
    x='edges',
    y='mst_time',
    markers=True,
    color='function',
    title='Edges vs MST Time by Function for Device 1',
    labels={'edges': 'Number of Edges', 'mst_time': 'MST Time (s)'}
)
fig6_3.update_xaxes(range=[0, max_edges])
fig6_3.update_yaxes(range=[0, max_total_time])
fig6_3.write_image(f"{plot_directory}edges_vs_mst_time_device_1.png")
fig6_3.show()

# Visualization 6.4: Line plot for `edges` vs `mst_time` grouped by function for device 2
fig6_4_data = device2_data.groupby(['function', 'dataset', 'num_clusters', 'vertices', 'edges'])[time_columns].mean().reset_index()
fig6_4_data = fig6_4_data.sort_values(by='edges')
fig6_4 = px.line(
    fig6_4_data,
    x='edges',
    y='mst_time',
    markers=True,
    color='function',
    title='Edges vs MST Time by Function for Device 2',
    labels={'edges': 'Number of Edges', 'mst_time': 'MST Time (s)'}
)
fig6_4.update_xaxes(range=[0, max_edges])
fig6_4.update_yaxes(range=[0, max_total_time])
fig6_4.write_image(f"{plot_directory}edges_vs_mst_time_device_2.png")
fig6_4.show()

# # Visualization 7: Scatter plot for `mst_time` vs `dataset`
# fig7 = px.scatter(
#     combined_data,
#     x='dataset',
#     y='mst_time',
#     color='device',
#     title='MST Time vs Dataset by Device',
#     labels={'dataset': 'Dataset', 'mst_time': 'MST Time (s)', 'device': 'Device'}
# )
# fig7.update_layout(xaxis_title='Dataset', yaxis_title='MST Time (s)')
# fig7.write_image(f"{plot_directory}mst_time_dataset_scatter.png")
# fig7.show()
#
# # Visualization 9: Heatmap for `edges`, `vertices`, and `mst_time`
# fig9 = px.density_heatmap(
#     average_data,
#     x='edges',
#     y='vertices',
#     z='mst_time',
#     title='Heatmap of MST Time by Edges and Vertices',
#     labels={'edges': 'Number of Edges', 'vertices': 'Number of Vertices', 'mst_time': 'MST Time (s)'},
#     color_continuous_scale='Viridis'
# )
# fig9.write_image(f"{plot_directory}heatmap_edges_vertices_mst_time.png")
# fig9.show()
#
# # Visualization 10: Scatter plot for `vertices` vs `total_time` grouped by dataset
# fig10 = px.scatter(
#     average_data,
#     x='vertices',
#     y='total_time',
#     color='dataset',
#     title='Vertices vs Total Time by Dataset',
#     labels={'vertices': 'Number of Vertices', 'total_time': 'Total Time (s)', 'dataset': 'Dataset'}
# )
# fig10.write_image(f"{plot_directory}total_time_vertices_scatter_plot.png")
# fig10.show()

# Visualization 11: Bar chart for total time across datasets
average_data_bar_chart = average_data_device.groupby(['dataset', 'device'])[time_columns].mean().reset_index()
fig11 = px.bar(
    average_data_bar_chart,
    x='dataset',
    y='total_time',
    color='device',
    barmode='group',
    title='Total Time Across Datasets by Device',
    labels={'dataset': 'Dataset', 'total_time': 'Total Time (s)', 'device': 'Device'}
)
fig11.write_image(f"{plot_directory}total_time_across_datasets.png")
fig11.show()

# Visualization 12: Bar chart for total time across datasets with devices and average of the devices
fig12_average_data_device = combined_data.groupby(['dataset', 'device'])[time_columns].mean().reset_index()
fig12_average_data_combined = combined_data.groupby(['dataset'])[time_columns].mean().reset_index()
fig12_average_data_combined['device'] = 'Average'

fig12_average_data_bar_chart = pd.concat(
    [
        fig12_average_data_device,
        fig12_average_data_combined
    ]
)
fig12 = px.bar(
    fig12_average_data_bar_chart,
    x='dataset',
    y='total_time',
    color='device',
    barmode='group',
    title='Total Time Across Datasets by Device and Average',
    labels={'dataset': 'Dataset', 'total_time': 'Total Time (s)', 'device': 'Device'}
)
fig12.write_image(f"{plot_directory}total_time_across_datasets_with_average.png")
fig12.show()

# Normalize total_time for slope comparison
min_total_time_pure_python = average_data[average_data['function'] == 'PURE_PYTHON']['total_time'].min()


def normalize_total_times(df, base_function):
    base_min_time = df[df['function'] == base_function]['total_time'].min()
    df['normalized_total_time'] = df['total_time'] - (base_min_time - min_total_time_pure_python)
    return df


normalized_total_time = average_data.groupby('function', group_keys=False).apply(
    lambda df: normalize_total_times(df, df['function'].iloc[0])
    )
normalized_total_time.sort_values(by='edges', inplace=True)

# Visualization 13: Line plot for `edges` vs normalized `total_time` grouped by function
fig13 = px.line(
    normalized_total_time,
    x='edges',
    y='normalized_total_time',
    markers=True,
    color='function',
    title='Edges vs Normalized Total Time by Function',
    labels={'edges': 'Number of Edges', 'normalized_total_time': 'Normalized Total Time (s)', 'function': 'Function'}
)
fig13.update_xaxes(range=[0, max_edges])
fig13.update_yaxes(range=[0, normalized_total_time['normalized_total_time'].max() * 1.05])
fig13.update_yaxes(title_text='Normalized Total Time (s)')
fig13.write_image(f"{plot_directory}edges_vs_normalized_total_time.png")
fig13.show()

# Visualization 13_1: Line plot for `edges` vs normalized `total_time` grouped by function with equal axis
fig13_1 = px.line(
    normalized_total_time,
    x='edges',
    y='normalized_total_time',
    markers=True,
    color='function',
    title='Edges vs Normalized Total Time by Function (Equal axis)',
    labels={'edges': 'Number of Edges', 'normalized_total_time': 'Normalized Total Time (s)', 'function': 'Function'}
)
fig13_1.update_xaxes(range=[0, max_edges])
fig13_1.update_yaxes(range=[0, max_total_time])
fig13_1.update_yaxes(title_text='Normalized Total Time (s)')
fig13_1.write_image(f"{plot_directory}edges_vs_normalized_total_time_equal_axis.png")
fig13_1.show()

# Normalize total_time for slope comparison
min_mst_time_pure_python = average_data[average_data['function'] == 'PURE_PYTHON']['mst_time'].min()


def normalize_mst_times(df, base_function):
    base_min_time = df[df['function'] == base_function]['mst_time'].min()
    df['normalized_mst_time'] = df['mst_time'] - (base_min_time - min_mst_time_pure_python)
    return df


normalized_mst_time = average_data.groupby('function', group_keys=False).apply(lambda df: normalize_mst_times(df, df['function'].iloc[0]))
normalized_mst_time.sort_values(by='edges', inplace=True)

# Visualization 14: Line plot for `edges` vs normalized `mst_time` grouped by function
fig14 = px.line(
    normalized_mst_time,
    x='edges',
    y='normalized_mst_time',
    markers=True,
    color='function',
    title='Edges vs Normalized MST Time by Function',
    labels={'edges': 'Number of Edges', 'normalized_mst_time': 'Normalized MST Time (s)', 'function': 'Function'}
)
fig14.update_xaxes(range=[0, max_edges])
fig14.update_yaxes(range=[0, normalized_mst_time['normalized_mst_time'].max() * 1.05])
fig14.update_yaxes(title_text='Normalized MST Time (s)')
fig14.write_image(f"{plot_directory}edges_vs_normalized_mst_time.png")
fig14.show()

# Visualization 14_1: Line plot for `edges` vs normalized `mst_time` grouped by function with equal axis
fig14_1 = px.line(
    normalized_mst_time,
    x='edges',
    y='normalized_mst_time',
    markers=True,
    color='function',
    title='Edges vs Normalized MST Time by Function (Equal axis)',
    labels={'edges': 'Number of Edges', 'normalized_mst_time': 'Normalized MST Time (s)', 'function': 'Function'}
)
fig14_1.update_xaxes(range=[0, max_edges])
fig14_1.update_yaxes(range=[0, max_total_time])
fig14_1.update_yaxes(title_text='Normalized MST Time (s)')
fig14_1.write_image(f"{plot_directory}edges_vs_normalized_mst_time_equal_axis.png")
fig14_1.show()

# Visualization 6.2: Line plot for `edges` vs `total_time` grouped by function and device
average_data = average_data.sort_values(by='num_clusters')
fig6 = px.line(
    average_data,
    x='num_clusters',
    y='total_time',
    markers=True,
    color='function',
    title='Number of Clusters vs Total Time by Function',
    labels={'num_cluster': 'Number of Clusters', 'total_time': 'Total Time (s)'}
)
fig6.update_xaxes(range=[0, max_num_clusters])
fig6.update_yaxes(range=[0, max_total_time])
fig6.write_image(f"{plot_directory}num_clusters_vs_total_time.png")
fig6.show()

# Prepare traces for total_time by function
total_time_traces = []
for function_name in average_data['function'].unique():
    function_data = average_data[average_data['function'] == function_name]
    total_time_traces.append(
        go.Scatter(
            x=function_data['num_clusters'],
            y=function_data['total_time'],
            mode='lines+markers',
            name=f'{function_name}'
        )
    )

# Prepare the trace for edges with enhanced style
edges_trace = go.Scatter(
    x=average_data['num_clusters'],
    y=average_data['edges'],
    mode='lines+markers',
    name='Edges',
    yaxis='y2',
    line=dict(
        width=4,  # Thicker line
        dash='dash',  # Dashed line
        color='red'  # Highlighted color
    ),
    marker=dict(
        size=10,  # Larger markers
        symbol='diamond'  # Distinct marker shape
    )
)

# Create the layout
layout = go.Layout(
    title='Number of Clusters vs Total Time with Edges by Function',
    xaxis=dict(
        title='Number of Clusters',
        range=[0, max_num_clusters]
    ),
    yaxis=dict(
        title='Total Time (s)',
        range=[0, max_total_time]
    ),
    yaxis2=dict(
        title='Edges',
        overlaying='y',
        side='right',
        range=[0, max_edges]
    ),
    legend=dict(
        title="Legend",
        orientation="h",
        x=0.5,
        xanchor="center",
        y=-0.2
    )
)

# Create the figure
fig = go.Figure(data=total_time_traces + [edges_trace], layout=layout)

# Save the figure and display
fig.write_image(f"{plot_directory}num_clusters_vs_total_time_with_edges_highlighted.png")
fig.show()
