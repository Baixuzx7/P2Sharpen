# P2Sharpen  
Code of P2Sharpen: A progressive pansharpening network with deep spectral transformation.  
  
#### running environment :<br>  
python=3.8, pytorch-gpu=1.7.1, matlab = 2018a.  
  
#### Preparation: <br>     
$\qquad$$\qquad$    [-] Construct the train, validation, test dataset according to the Wald protocol.  
$\qquad$$\qquad$    [-] Put the all the dataset in the root directory, namely TrainFolder, ValidFolder and TestFolder.  
$\qquad$$\qquad$    [-] In each directory, there are four subdirectories, namely pan_label/ ms_label/ pan/ ms/  
$\qquad$$\qquad$    [-] The images in each directory should correspond to each other.  

#### To train :<br>    
$\qquad$$\qquad$    [-] The whole training process contains two part, STNet and P2Net.  
$\qquad$$\qquad$    [-] Please run "transfertrain.py" to learn the spectral tranformation network(STNet).  
$\qquad$$\qquad$    [-] TNet guides the optimization of P2Net, so ensuring the accuracy before the next step.  
$\qquad$$\qquad$    [-] Please run "fusiontrain.py" to learn the progressive pansharpening network (P2Net).  

#### To valid :<br>    
$\qquad$$\qquad$    [-] Use the functions in the file ".\Eval.py" or others to evalute the performance on valid dataset.  
$\qquad$$\qquad$    [-] Pick out the best parameters and save it in path "./Model/P2Net/fusion.pth".  
  
#### To test :<br>  
$\qquad$$\qquad$    [-] Running the "fusionpredict.py" to generate the pansharpening results.  
$\qquad$$\qquad$    [-] Open the Matlab and running the file ".\Evalution\FusionEval.m".  
      
