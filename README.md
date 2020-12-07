
## AICS Features Extraction

### from aicsfeatures.extractor import *

#### /feature_calc

Functions related to feature extraction. Feature extraction should be done like this:

```
from aicsfeatures.extractor import *

features_result = xxx.GetFeatures(args=None, seg=image_xxx)
```

where xxx can be

* **mem**, for cell membrane-related features
* **dna**, for nucleus-related features
* **structure**, for structure-specific features.
* **stack**, for stack-related features.

Assumptions of each function about the input image should be detailed inside the function like this

```
    Assumptions:
        - Input is a 3D 16-bit image
        - There is a single object of interest
        - Background has value 0
        - Object of interest have pixel value > 0
```

*The result `features_result` should always be a single row Pandas dataframe.*

**exutils.py:** Main routine for feature calculation. We try to re-use functions from skimage (shape analysis) and Mahotas (texture analysis). Here is also the place to implement new features that are not found in those packages. We try to be general and do not say anything specific about the type of biological structure that the input image should represent.

**mem.py:** Wrappers for feature extraction of cell membrane images.

**dna.py:** Wrappers for feature extraction of dna images.

**structure.py:** Wrappers for feature extraction of structures images. Things to be defined: which type of feature we should extract for each given structure. Or do we do all to all?

**stack.py:** Wrappers for feature extraction of whole stack images.
