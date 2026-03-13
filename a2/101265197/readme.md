### COMP4900 W26 A2

### Tom Fan - 101 265 197

#### Notes for grader/TA:

- The project was built using Python 3.13.x - Python 3.12 and above should work, but for guaranteed compatibility I would suggest
using Python 3.13
- There is a mismatch between the indicated vertex counts in the SNAP reddit and Wikipedia datasets. This is likely because
the normalization methods used by the authors are different from what I'm using. 
  - There are some entries in both datasets that are blank, which I have normalized away. This suggests that extra cleaning
    on the dataset is in order, but not necessarily in the scope of this assignment as we are just demonstrating use and understanding
    of centrality measures.
- The relative imports of the graph files in my scripts assume the user will be running the script inside the project directory
  itself.

I went ahead and included the venv with the submission, hopefully that removes any further ambiguity in versioning. If
it does not work, try wiping it and creating a new one.