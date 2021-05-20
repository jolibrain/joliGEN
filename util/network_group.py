class NetworkGroup:
  def __init__(self, networks_to_optimize, networks_not_to_optimize,forward_functions,backward_functions,loss_names_list,optimizer,loss_backward):
    self.networks_to_optimize = networks_to_optimize
    self.networks_not_to_optimize = networks_not_to_optimize
    self.forward_functions = forward_functions
    self.backward_functions = backward_functions
    self.loss_names_list = loss_names_list
    self.optimizer = optimizer
    self.loss_backward = loss_backward
