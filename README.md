# FairU

A study of Metric and Framework Improving Fairness-utility Trade-off in Link Prediction

### **Overall Framework of Logit Averaging**
<img src=./assets/LA_framework.jpg>


## Setups
[![Python](https://img.shields.io/badge/python-3.9.12-blue?logo=python&logoColor=FED643)](https://www.python.org/downloads/release/python-385/)
[![Pytorch](https://img.shields.io/badge/pytorch-1.13.0-red?logo=pytorch)](https://pytorch.org/get-started/previous-versions/)


## Datasets
The dataset name must be specified in the "--dataset" argument
- [Diginetica](https://competitions.codalab.org/competitions/11161#learn_the_details-data2)
- [Retailrocket](https://www.kaggle.com/retailrocket/ecommerce-dataset)
- [Yoochoose 1/64](https://www.kaggle.com/chadgostopp/recsys-challenge-2015) (using latest 1/64 fraction due to the amount of full dataset)
- [Tmall](https://ijcai-15.org/repeat-buyers-prediction-competition/)

## Train and Test
Run `main.py` file to train the model. You can configure some training parameters through the command line.
```
python main.py
```


## Citation
Please cite our paper if you use the code:
```
@article{yang2022la,
  title={Logit Averaging: Capturing Global Relation for
Session-Based Recommendation},
  author={Heeyoon Yang, Gahyung Kim, Jee-Hyong Lee},
  journal={Applied Science - Special Issue},
  year={2022},
  doi={10.3390/app12094256}
}
```

## Reference 
1) [NARM: Neural Attentive Session-based Recommendation](https://github.com/Wang-Shuo/Neural-Attentive-Session-Based-Recommendation-PyTorch)
2) [EOPA of LESSR: Handling Information Loss of Graph Neural Networks for Session-based Recommendation](https://github.com/twchen/lessr)
3) [NISER: Normalized Item and Session Representations to Handle Popularity Bias](https://github.com/johnny12150/NISER)
4) [SRGNN: Session-based Recommendation with Graph Neural Network](https://github.com/CRIPAC-DIG/SR-GNN)
5) [SRSAN: Session-based Recommendation with Self-Attention Networks](https://github.com/GalaxyCruiser/SR-SAN)
6) [TAGNN++: Introducing Self-Attention to Target Attentive Graph Neural Networks](https://github.com/The-Learning-Machines/SBR)

