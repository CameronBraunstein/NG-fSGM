python main.py --algorithm fSGM --penalty_model Potts --dataset Middlebury\
  --scene RubberWhale --displacement_window 11 --visualize_flow --save_cost_tensor\
  --save_flow_file
#In the original paper, a truncated linear model is chosen. However,
#Their provided equations look as if there is a typo (Eq.(13)). I will investigate 
#their cited source paper for the truth.