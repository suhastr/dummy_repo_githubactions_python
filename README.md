# arithmetic

A dummy Python project for GitHub Actions CI/CD demo.

This project uses the standard Python build configuration (`pyproject.toml`) and is managed using **Poetry** for dependency resolution and virtual environments.

## 🚀 Getting Started

Follow these steps to set up the project and install all necessary dependencies, including development tools.

### Prerequisites

You must have **Poetry** installed on your system. If you don't have it, you can find installation instructions on the [Poetry website](https://python-poetry.org/docs/#installation).

### Installation

1.  **Clone the Repository**

    ```bash
    git clone <your-repository-url>
    cd arithmetic
    ```

2.  **Install Dependencies**

    Use the following command to install the project's runtime dependencies (`numpy`) and all defined optional groups, specifically the `dev` group, which includes all testing and development tools.

    ```bash
    poetry install --extras dev
    ```

    This command will:
    * Create or use a dedicated virtual environment.
    * Install the main dependency: `numpy`.
    * Install all development dependencies: `pytest`, `black`, `ruff`, `mypy`, `pylint`, and `pytest-cov`.
    * Create a `poetry.lock` file for reproducible builds.

## 🛠 Usage

### Running Code

To execute any Python script within the project's isolated environment, use the `poetry run` prefix:

```bash
poetry run python src/main.py