# Context-attended-Graph-Convolution-Network

[Paper](https://doi.org/10.1016/j.patrec.2020.02.006) | [TensorFlow](https://www.tensorflow.org/)

Context-attended Graph Convolution Network for Remote Sensing Images

<img src=edge_att_r1.png alt="Proposed edge attention." width="700">


This set of codes implementscontext-attended (edge attention + node attention) single-label classification/retrieval of images using graph convolution network (GCN). In this work, it is implemented for VHR airborne images using the <a href="http://bigearth.eu/assets/docs/multilabels.zip">single-label annotated UCMERCED dataset</a>  (source: <a href="https://ieeexplore.ieee.org/document/8089668/">Chaudhuri et al</a>) and PATTERNNET dataset, but it is a generic framework. The GCN used in this framework is inpired from <a href="https://ieeexplore.ieee.org/document/7979525/">Such et al</a>. The codes are written using the TensorFlow framework (version 1.2.1) in Python 2.7.12. The input needed for the code is adjacency matrices of the graph, node features and label set. 

To implement the code:
<ol>
<li>Check the gpu being used and amount of gpu in <b>src/graphcnn/experiment_multilabel.py</b> file (look for this code snippet in the _init_ function, here 0.2 fraction of gpu0 is being used) <br>
 &nbsp &nbsp os.environ["CUDA_VISIBLE_DEVICES"] = '0’ <br>
 &nbsp &nbsp self.config.gpu_options.per_process_gpu_memory_fraction = 0.2 
 </li> 

<li>If needed change the path of snapshots and summary folders in ‘run’ function of <b>src/graphcnn/experiment_multilabel.py</b> by changing the ‘path’ variable

<li> Change the mat file locations in <b>src/graphcnn/setup/ucmerced.py</b> (the mat files holding the adjacency matrices and node features) i.e.   <br> 
     &nbsp &nbsp dataset= scipy.io.loadmat('/path_to_mat_file/new_dataset.mat')</li>


<li> While in src folder, run the <b>run_graph.py</b> file (for terminal based, type ‘python run_graph.py’ in terminal) </li> </ol>

The various useful files and their details are:
<ol>
<li> <b> src/graphcnn/setup/ucmerced.py </b> - the file which loads the mat file and sends it to run_graph.py which is used later. You need to load the graph’s adjacency matrix, features and training labels here. </li>

<li> <b> src/run_graph.py </b> - the file to be run, from which load_ucmerced_data, preprocess_data and experiment_multilabel functions are called. Edit this if you want to change architecture and other misc parameters.</li>

<li> <b> src/graphcnn/experiment_multilabel.py </b> - the main file, having the main functions called from run_graph.py. Edit the various functions, main function is run_experment from which others are called like create_test_train, create_data and run. </li>

<li> <b> src/graphcnn/network.py, Graph-CNN/src/graphcnn/layer.py </b> - the back end files containing the details about graph cnn layers. DO NOT edit unless you want to make any change in architecture </li> 

<li> <b> src/multi_label.py</b> - to run the multi label comparison experiments with SVC, MLkNN and GaussianNB. Loads GCN’s train, test and val data and uses the same here.</li>
</ol>

<img src=node_att_r1.png alt="Existing node attention." width="700">

The new_dataset.mat and features.mat files can be downloaded from this <a href="https://www.dropbox.com/sh/k9ugragg6fbej6n/AAD6g6lhZm4243g0plTKaPJza?dl=0&m="> Dropbox </a> link. 


### Paper

*    The paper is also available at: [Siamese graph convolutional network for content based remote sensing image retrieval](https://reader.elsevier.com/reader/sd/pii/S1077314219300578?token=FF18DF6BD33340CB07179AE964A960F224B8A29AC597C8D4875C71AF366407364D877984BA4E1BA4FF97548B3C83FB2A)

*   Feel free to cite the author, if the work is any help to you:

```
@article{chaudhuri2019siamese,
  title={Siamese graph convolutional network for content based remote sensing image retrieval},
  author={Chaudhuri, Ushasi and Banerjee, Biplab and Bhattacharya, Avik},
  journal={Computer Vision and Image Understanding},
  volume={184},
  pages={22--30},
  year={2019},
  publisher={Elsevier}
}


