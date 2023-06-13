# Use the official Python 3.9 image as the base image
FROM python:3.9


# Copy the source code to the container
COPY . .

# Install pauliopt
RUN python setup.py install

# Run the dependency test (all imports should work)
RUN python ./dependency_test.py

# Install the Python dependencies for testing (we require pytket, qiskit, ...)
# RUN pip install --no-cache-dir -r requirements.txt


# Execute all the unit tests in the ./tests folder
CMD ["python", "-m", "unittest", "discover", "-s", "./tests/", "-p", "test_*.py"]