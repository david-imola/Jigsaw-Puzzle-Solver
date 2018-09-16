#Run this before commiting
find . -name __pycache__ -type d -exec rm -rf {} \;
find . -name *.pyc -type f -exec rm -f {} \;