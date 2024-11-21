# Makefile for project setup and maintenance
CONDA_INSTALLER_URL = https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

all: check_conda validate

# Install conda if not already installed
check_conda:
	@if ! command -v conda &> /dev/null; then \
		echo "Conda not found. Installing Miniconda..."; \
		curl -o miniconda-installer.sh $(CONDA_INSTALLER_URL); \
		chmod +x miniconda-installer.sh \
		./miniconda-installer.sh-b -p $$HOME/miniconda; \
		rm miniconda-installer.sh \
		echo "export PATH=\$$HOME/miniconda/bin:\$$PATH" >> $$HOME/.bashrc; \
		echo "Conda installation complete. Restart your shell or source ~/.bashrc to use Conda."; \
	else \
		echo "Conda is already installed."; \
	fi

# Create a Conda environment using environment.yml
env:
	conda env create -f environment.yml --prefix ./.conda

# Format code with Black and isort
format:
	conda activate ./.conda && black . && isort .

# Lint code using Flake8 and Pydocstyle
lint:
	conda activate ./.conda && flake8 . && pydocstyle .

# Run tests
test:
	conda activate ./.conda && pytest

# Run all linting and formatting checks
polish:
	conda activate ./.conda && black . && isort . && flake8 . && pydocstyle .

# Run all linting and formatting checks and tests 
validate:
	conda activate ./.conda && black --check . && isort --check . && flake8 . && pydocstyle .  && pytest .