# Copilot Instructions - Delivery Time Analysis

## Project Overview
Data analysis project focused on delivery time analysis and visualization. Currently in early development stage with a single analysis script.

## Environment & Dependencies
- **Core libraries**: pandas, numpy for data manipulation
- **Visualization**: matplotlib, seaborn for plotting
- **Standard library**: datetime for time-based operations
- Python file naming: Snake_case without `.py` extension (e.g., `delivery_time`)

## File Structure
- `delivery_time`: Main analysis script (Python without `.py` extension)
- Analysis scripts follow executable-style naming convention

## Development Workflow
1. **Python Environment**: Ensure data science stack is installed
   ```powershell
   pip install pandas numpy matplotlib seaborn
   ```

2. **Running Scripts**: Execute directly as Python files
   ```powershell
   python delivery_time
   ```

## Coding Conventions
- **Import order**: Standard library → Data manipulation (pandas, numpy) → Visualization (matplotlib, seaborn) → Datetime utilities
- Use pandas for data loading and transformation
- Use seaborn as primary plotting library for statistical visualizations
- Follow data science workflow: load → clean → analyze → visualize

## Data Analysis Patterns
- Expected data: Delivery-related datasets with timestamps
- Time-based analysis using datetime module
- Statistical visualizations with seaborn's default styling

## AI Agent Guidance
- Assume this is a data exploration and visualization project
- When adding analysis code, follow EDA (Exploratory Data Analysis) structure
- Include docstrings for analysis functions
- Add inline comments for complex data transformations
- Visualization outputs should be clear and labeled (titles, axis labels, legends)
