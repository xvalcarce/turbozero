import pandas as pd
import matplotlib.pyplot as plt

### Firstplot success / depth ###

# Load the data
data = pd.read_csv('data/benchmark.csv')

# Set up the figure and the primary axis with a square shape
fig, ax1 = plt.subplots(figsize=(7, 7))

# Define the bar width
bar_width = 0.4

# Plot compilation success data on the primary axis (left y-axis) as stacked bars
bars1 = ax1.bar(data['depth'] - bar_width/2,
        data['az_deterministic'],
        width=bar_width,
        label='AZ Deterministic',
        color='#7493d2')

bars2 = ax1.bar(data['depth'] - bar_width/2,
        data['az_stochastic'],
        width=bar_width,
        bottom=data['az_deterministic'],
        label='AZ Stochastic',
        color='#b4befe')

bars3 = ax1.bar(data['depth'] + bar_width/2,
        data['mcts_deterministic'],
        width=bar_width,
        label='MCTS Deterministic',
        color='#d27e46')

bars4 = ax1.bar(data['depth'] + bar_width/2,
        data['mcts_stochastic'],
        width=bar_width,
        bottom=data['mcts_deterministic'],
        label='MCTS Stochastic',
        color='#fab387')

# Set labels for the primary axis with bigger font size
ax1.set_xlabel('Depth', fontsize=14, color='#2d3748')
ax1.set_ylabel('Compilation Success', fontsize=14, color='#2d3748')
ax1.set_xticks(data['depth'])
ax1.tick_params(axis='y', labelsize=12, color='#2d3748')

# Create a secondary axis for the average compiled depth
ax2 = ax1.twinx()

# Plot average compiled depth data on the secondary axis (right y-axis) only if non-zero
non_zero_az_avg_depth = data['az_avg_depth'][data['az_avg_depth'] != 0]
non_zero_mcts_avg_depth = data['mcts_avg_depth'][data['mcts_avg_depth'] != 0]

depths_az = data['depth'][data['az_avg_depth'] != 0]
depths_mcts = data['depth'][data['mcts_avg_depth'] != 0]

color2 = '#a6d189'
line1, = ax2.plot(depths_az, non_zero_az_avg_depth, label='AZ Avg Depth', color=color2, linestyle='-', marker='^')
line2, = ax2.plot(depths_mcts, non_zero_mcts_avg_depth, label='MCTS Avg Depth', color=color2, linestyle='-', marker='o')

# Set labels for the secondary axis with bigger font size
ax2.set_ylabel('Average Compiled Depth', fontsize=14, color=color2)
ax2.tick_params(axis='y', labelsize=12, color=color2)

# Set the grid to be in the background
ax1.set_axisbelow(True)
ax2.set_axisbelow(True)

# Create a single legend combining both sections
ax1.legend(handles=[bars1, bars2, bars3, bars4, line1, line2],
          labels=['AZ Deterministic', 'AZ Stochastic', 'MCTS Deterministic', 'MCTS Stochastic', 'AZ Avg Depth', 'MCTS Avg Depth'],
          loc='center right', facecolor='white')

# Use LaTeX for rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

# Adjust layout to make room for the legend and reduce whitespace
plt.tight_layout()
plt.subplots_adjust(left=0.1, right=0.85)

# Save the plot to a PDF file
pdf_path = 'plots/benchmark.pdf'
plt.savefig(pdf_path, format='pdf', bbox_inches='tight')

### Second plot success / number of iterations ###

data = pd.read_csv('data/niters.csv')

fig, ax1 = plt.subplots(figsize=(7, 7))

# Plot success rate on left y-axis
color = '#7493d2'
ax1.set_xlabel('Number of Iterations',fontsize=14)
ax1.set_ylabel('Success Rate', color=color, fontsize=14)
ax1.plot(data['niters'], data['success'], marker='o', color=color, label='Success Rate')
ax1.tick_params(labelsize=14)
ax1.tick_params(axis='y',color=color,labelsize=14)

# Create a second y-axis for time
ax2 = ax1.twinx()
color = '#d27e46'
ax2.set_ylabel('Time (s)', color=color, fontsize=14)
ax2.plot(data['niters'], data['time'], marker='s', linestyle='--', color=color, label='Time')
ax2.tick_params(axis='y',color=color,labelsize=14)

# Set the grid to be in the background
ax1.set_axisbelow(True)
ax2.set_axisbelow(True)

# Use LaTeX for rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

# Adjust layout to make room for the legend and reduce whitespace
plt.tight_layout()
plt.subplots_adjust(left=0.1, right=0.85)

# Save the plot to a PDF file
pdf_path = 'plots/niters.pdf'
plt.savefig(pdf_path, format='pdf', bbox_inches='tight')

