.PHONY: install clean test export_env

# Install the package
install: 
	python setup.py install 

# Clean the package
clean:	
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache 
	
# Run all tests in tests/
test:
	cd tests
	pytest tests/*.py

# Export the environment to the yaml file
export_env:
	conda env export > env.yaml
