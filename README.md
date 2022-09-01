# P2Sharpen  
Code of P2Sharpen: A progressive pansharpening network with deep spectral transformation.  
  
#### Running environment :<br>  

python=3.8, pytorch-gpu=1.7.1, matlab = 2018a.  
  
#### Preparation: <br>  

* Construct the train, validation, test dataset according to the Wald protocol.  <br> 
* Put the all the dataset in the root directory, namely TrainFolder, ValidFolder and TestFolder. <br>  
* In each directory, there are four subdirectories, namely pan_label/ ms_label/ pan/ ms/  <br> 
* The images in each directory should correspond to each other.  <br>

#### To train :<br>    

* The whole training process contains two part, STNet and P2Net.  <br>
* Please run "transfertrain.py" to learn the spectral tranformation network(STNet).  <br>
* TNet guides the optimization of P2Net, so ensuring the accuracy before the next step.  <br>
* Please run "fusiontrain.py" to learn the progressive pansharpening network (P2Net).  <br>

#### To valid :<br>    

* Use the functions in the file ".\Eval.py" or others to evalute the performance on valid dataset. <br>
* Pick out the best parameters and save it in path "./Model/P2Net/fusion.pth".  <br>

#### To test :<br>  

* Run the "fusionpredict.py" to generate the pansharpening results.  <br>
* Open the Matlab and run the file ".\Evalution\FusionEval.m".  <br>
