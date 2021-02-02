VENV = Udacity_RL_P1
INTERPRETER = $(shell which python3)

all: 
	@create_venv
	@echo Virtual env created!
	@install
	@echo Requirements installed and Jupyter kernel created!

create_venv:
	pip install virtualenv
	virtualenv -p $(INTERPRETER) $(VENV)
	

install: SHELL:=/bin/bash
install:
	( \
       source ./$(VENV)/bin/activate; \
       pip install -r requirements.txt; \
	   python -m ipykernel install --user --name=$(VENV) \
    )

phony:
	all create_venv install