# P2Sharpen
Code of P2Sharpen: A progressive pansharpening network with deep spectral transformation.

#### running environment :<br>
python=3.8, pytorch-gpu=1.7.1, matlab = 2018a.

#### Prepare data :<br>
Preparation: 
    [-] Construct the train, validation, test dataset according to the Wald protocol.
    [-] Put the all the dataset in the root directory, namely TrainFolder, ValidFolder and TestFolder.
    [-] In each directory, there are four subdirectories:
        ../TrainFolder/
                      pan_label/
                      ms_label/
                      pan/
                      ms/
    [-] The images in each directory should correspond to each other.

#### To train :<br>
Train:
    [-] The whole training process contains two part, STNet and P2Net.
    [-] Please run "transfertrain.py" to learn the spectral tranformation network(STNet).
    [-] TNet guides the optimization of P2Net, so ensuring the accuracy before the next step.
    [-] Please run "fusiontrain.py" to learn the progressive pansharpening network (P2Net).

Valid:
    [-] Use the functions in the file ".\Eval.py" or others to evalute the performance on valid dataset.
    [-] Pick out the best parameters and save it in path "./Model/P2Net/fusion.pth".

#### To test :<br>
Test:
    [-] Running the "fusionpredict.py" to generate the pansharpening results.
    [-] Open the Matlab and running the file ".\Evalution\FusionEval.m".
    
