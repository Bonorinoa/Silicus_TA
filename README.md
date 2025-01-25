# Silicus_TA

## Overview

Silicus_TA is a repository for the Silicus Teaching Assistant (TA) application. This application is designed to assist students in ECON 57 and ECON 101 courses by providing a conversational interface to interact with course materials. The main features of Silicus_TA include generating explanations, creating exercises, and providing clarifications on course concepts.

![Demo Screenshot](C:\Users\Bonoc\Desktop\CGU_Semesters\FALL_2023\ECON57\Silicus_TA\demo_image.png)

## Features

- **Conversational Interface**: Interact with the course material in a conversational manner.
- **Exercise Generation**: Generate exercises related to specific assignments, concepts, or lectures.
- **Clarifications**: Get clarifications on learning outcomes and other general inquiries about the course.
- **Data Source Finder**: Find relevant data sources for economics projects based on your query.

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Bonorinoa/Silicus_TA.git
   cd Silicus_TA
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv silicus_ta_venv
   source silicus_ta_venv/bin/activate  # On Windows use `silicus_ta_venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the Streamlit application**:
   ```bash
   streamlit run silicus_ta.py
   ```

2. **Access the application**:
   Open your web browser and go to `http://localhost:8501`.

## Directory Structure

- `ECON101_Files/`: Contains TeX files related to the ECON 101 course.
- `ECON57_Files/`: Contains TXT files related to the ECON 57 course.
- `pages/`: Contains the Streamlit pages for the application.
- `silicus_ta_venv/`: Virtual environment directory.
- `utils.py`: Utility functions for the application.
- `tex_loader.py`: Custom loader for TeX files.
- `requirements.txt`: List of dependencies for the project.
- `README.md`: This file.

## Examples of Interaction

### Example 1: Generating Exercises

**User**: Generate more exercises related to Lecture 3.

**Silicus_TA**: Here are some additional exercises related to Lecture 3:
1. Explain the concept of standard deviation and how it is used in statistics.
2. Create a frequency distribution table for the given data set.
3. Calculate the mean, median, and mode for the following data set: [5, 10, 15, 20, 25].

### Example 2: Clarifying Concepts

**User**: Explain the difference between nominal and real variables.

**Silicus_TA**: Nominal variables are measured in current prices without adjusting for inflation, while real variables are adjusted for changes in the price level, reflecting true purchasing power. For example, nominal GDP is measured in current prices, whereas real GDP is adjusted for inflation.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
