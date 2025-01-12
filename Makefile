.PHONY: install clean test

install: 	
	python setup.py install 

clean:
	rm -rf build dist *.egg-info
	rm -rf .pytest_cache 
	
test:
	cd tests
	pytest tests/*.py

: 
