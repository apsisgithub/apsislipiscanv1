# apsislipiscanv1
![](/usage/images/apsis.png) 

Apsis-Lipiscanv1 is an ocr system for Printed Bangla Documents developed at [Apsis Solutions limited](https://apsissolutions.com/)

The full system is build with 2 components: 
* Text detection : DBNet
* Text recognition:ApsisNet
    
# **Installation**

* **create a conda environment**: 

```bash
conda create -n apsislipiscanv1 python=3.9
```

* **activate conda environment**: 

```bash
conda activate apsislipiscanv1

```

## **As module/pypi package**
### **cpu installation**

```bash
pip install apsisocr
pip install onnxruntime
pip install fastdeploy-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
pip install scikit-learn
pip install numpy==1.23.0
```

### **gpu installation**

It is recommended to use conda environment . Specially for GPU.

* **installing cudatoolkit and cudnn**: 

```bash
conda install cudatoolkit
conda install cudnn
# conda install cudatoolkit=11.3.1 -c nvidia <--- install this if loading gives issue
```

* **installing packages**

```bash
pip install apsisocr
pip install onnxruntime-gpu==1.16.0
python -m pip install -U fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
pip install scikit-learn
pip install numpy==1.23.0
```

* **exporting environment variables**

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```


# Useage


## Apsisnet : Bangla Recognizer

* useage
```python
from apsisocr import ApsisNet
bnocr=ApsisNet()
bnocr.infer(crops)
```
* docstring for ```ApsisNet.infer```

```python
"""
Perform inference on image crops.

Args:
    crops (list[np.ndarray]): List of image crops.
    batch_size (int): Batch size for inference (default: 32).
    normalize_unicode (bool): Flag to normalize unicode (default: True).

Returns:
    list[str]: List of inferred texts.
"""
```


## PaddleDBNet : Text Detector

* check [paddleOCR](https://github.com/PaddlePaddle/PaddleOCR) official website for better understanding of the model

```python
# initialization
from apsisocr import PaddleDBNet
detector=PaddleDBNet()
# getting word boxes
word_boxes=detector.get_word_boxes(img)
# getting line boxes
line_boxes=detector.get_line_boxes(img)
# getting crop with either of the results
crops=detector.get_crops(img,word_boxes)
```




# **Deployment**
* running the app:

```bash
streamlit run app.py --server.port 3033 # deploys streamlit built frontend at 3033 port
```

![](/usage/images/ss1.png) 

* The **app.py** runs a streamlit UI 

![](/usage/images/ss2.png) 


# DB setup

```bash
pip install sqlalchemy
conda install -c conda-forge psycopg2
pip install loguru
```



# License
Contents of this repository are restricted to non-commercial research purposes only under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/). 

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>

