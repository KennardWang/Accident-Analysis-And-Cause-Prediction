# Accident-Analysis-And-Cause-Prediction

> Kennard acquired some practical knowledge of machine learning through the online course, and now he cannot wait to 
> complete a data analysis task assigned by his boss. This task is about car accidents and it comprised of
> rudimentary data plotting and main cause prediction.

------

### Demo
+ Overall Data

![demo1](https://kennardwang.github.io/ImageSource/Accident-Analysis-And-Cause-Prediction/demo1.png)

+ Density In District Dimension

![demo2](https://kennardwang.github.io/ImageSource/Accident-Analysis-And-Cause-Prediction/demo2.png)

+ Density In Time Dimension

![demo3](https://kennardwang.github.io/ImageSource/Accident-Analysis-And-Cause-Prediction/demo3.png)

+ Prediction With NN Forward Propagate

![demo4](https://kennardwang.github.io/ImageSource/Accident-Analysis-And-Cause-Prediction/demo4.png)

------

### Development Environment

| Description | Specification |
|:---:|:---:|
| System | Windows 10 |
| Language | Python 3.7 ( Anaconda ) |
| IDE | PyCharm 2020.2.2 ( Community Edition ) |

------

### Data Specification

|Column|Range|Description|
|:---:|:---:|:---:|
|District|1~12|12 different districts in our city|
|Time|0~23|24 hours|
|Cause|1~4|4 different main causes: 1 = collision with another car; 2 = collision with bicycles or pedestrian; 3 = collision with construction materials such as building, road facilities, trees and animals; 4 = extremely bad weather and unpredicted accidents|

------

### Contribution
+ Pre-process data by Excel, for the sake of extracting discrete features.
+ Analyze the relationship among districts, time and main causes, as well as plot the density in both district and time dimension.
+ Build 2 models ( Logistic Regression and Neural Netowrk ) to predict the main cause of a car accident.

------

### License
+ [MIT License](https://github.com/KennardWang/Accident-Analysis-And-Cause-Prediction/blob/master/LICENSE)

------

### Author
+ Kennard Wang ( 2021.8.5 )
