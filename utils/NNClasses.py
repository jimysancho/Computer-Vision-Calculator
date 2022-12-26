from .functions import *

class Layer:
    
    def __init__(self, activation, deriv_activation, nodes, batch_normalization=False):
        """
        activation: function. 
            - Function that computes the output of the neuron. 
            - It is used in forward propagation. 
            
        deriv_activation: function. 
            - Function that computes the derivative of the activation function. 
            - It is used in backpropagation. 
            
        nodes: int
            - Number of nodes in the layer. 
        """
        self.activation = activation
        self.deriv_activation = deriv_activation
        self.nodes = nodes
        self.batch_normalization = batch_normalization
        
        self.__output = None
        self.__delta = None
        self.__input = None
        self.__init = False
        
        self.__regularization_functions = {'l2': self.__l2, 'l1': self.__l1}
        
        self.__v_w = 0
        self.__s_w = 0
        
        self.__v_b = 0
        self.__s_b = 0
        
        self.__epsilon = 1e-9
        
    def random_init(self, input_shape):
        """
        input_shape: int
            - Number of inputs going into the layer (outputs of the layer before this one). 
        """
        
        if not self.__init:
            self.W_ = np.random.randn(self.nodes, input_shape) * 1 / np.sqrt(input_shape)
            self.b_ = np.zeros((self.nodes, 1))
            self.__init = True
            
            if self.batch_normalization:
                self.gamma_ = np.random.randn(self.nodes, 1)
                self.beta_ = np.zeros((self.nodes, 1))
    
    def forward(self, back_input):
        """
        back_input: array of shape: [output_{l - 1}, 1] 
            - (l - 1) refers to the layer before self. Its output is the new input. 
            
        batch_normalization: bool 
            - If True, batch normalization is applied. 
        """
        
        assert self.__init, 'You must initialize the weights first'
        
        Z = self.W_.dot(back_input) + self.b_
        
        if self.batch_normalization:
            Z_norm = (Z - Z.mean()) / (Z.std() + self.__epsilon)
            Z = Z_norm * self.gamma_ + self.beta_
            
        self.__output = self.activation(Z)
        
        self.__input = back_input.copy()
        self.cache_ = {}
        
        self.cache_['Z'] = Z if not self.batch_normalization else Z_norm
        self.cache_['W'] = self.W_
        self.cache_['b'] = self.b_
                
    def backward(self, forward_delta, forward_weights):
        """
        forward_delta: array. shape = [n_units_{l+1}, 1]
            - (l + 1) refers to the number of units of the layer next to self. 
        forward_weights: array. shape = [n_units_{l+1}, self.nodes]
            - The weights of the forward layer
            
        This function computes the backward step: it computes the error of the layer. 
        """
        self.__delta = forward_weights.T.dot(forward_delta) * self.deriv_activation(self.cache_['Z'])
        
    def update_weights(self, eta, last_layer=False, t=0, 
                       grad=None, **kwargs):
        """
        eta: float. 
            - Learning rate.
            
        last_layer: bool
            - It this is the last layer of the deep neural networ, we need to compute delta 
              in order to begin the gradient descent through the neuron. 
              
        grad: array-like. Optional. 
            - To compute the error of the last layer

        t: int. 
            - Epoch of the learning process
          
        """
        
        self.__eta = eta
        self.__t = t
        try:
            algorithm = kwargs['algorithm']
        except:
            algorithm = 'mini_batch'
            
        try:
            regularization = kwargs['regularization']
        except:
            regularization = None
            
        if last_layer:
            assert grad is not None, 'Insert the gradient to compute the last layer error: delta'
            self.__delta = grad * self.deriv_activation(self.cache_['Z'])
        
        grad_W = self.__delta.dot(self.__input.T) 
        grad_b = np.sum(self.__delta, axis=1).reshape((-1, 1))
        
        if self.batch_normalization:
            grad_gamma = self.__delta * self.cache_['Z']
            grad_beta = grad_b.copy()
            
            self.gamma_ -= eta * grad_gamma
            self.beta_ -= eta * grad_beta
        
        if regularization is not None:
            regularization, l = regularization
            assert regularization in ('l2', 'l1', 'dropout'), 'Regularization not valid'
            regularization_function = self.__regularization_functions[regularization]
            regularization_function(l, grad_W)
            
        if algorithm == 'mini_batch':
            self.W_ -= eta * grad_W
            self.b_ -= eta * grad_b
            return 
        
        else:
            algorithm, betas = algorithm[0], algorithm[1:]
            assert algorithm in ('rmsprop', 'adam', 'momentum'), 'Not valid algorithm to update the parameters'
        
        if algorithm == 'adam':
            grad_W, grad_b = self.__adam_update(grad_W, grad_b, betas[0], betas[1])
            
        elif algorithm == 'rmsprop':
            grad_W, grad_b = self.__rmsprop_update(grad_W, grad_b, betas[1])
            
        elif algorithm == 'momentum':
            grad_W, grad_b = self.__momentum_update(grad_W, grad_b, betas[0])
        
        self.W_ -= eta * grad_W
        self.b_ -= eta * grad_b 
            
    def __rmsprop_update(self, grad_W, grad_b, beta2):
        self.__compute_s(grad_W, grad_b, beta2)
        grad_W = grad_W / (np.sqrt(self.__s_w) + self.__epsilon)
        grad_b = grad_b / (np.sqrt(self.__s_b) + self.__epsilon)
        return grad_W, grad_b
        
    def __momentum_update(self, grad_W, grad_b, beta1):
        self.__compute_v(grad_W, grad_b, beta1)
        grad_W = self.__v_w.copy()
        grad_b = self.__v_b.copy()
        return grad_W, grad_b
    
    def __adam_update(self, grad_W, grad_b, beta1, beta2):
        self.__compute_v(grad_W, grad_b, beta1)
        self.__compute_s(grad_W, grad_b, beta2)
        
        grad_W *= (np.sqrt(1.0 - beta2 ** self.__t) / (1.0 - beta1 ** self.__t))
        grad_b *= (np.sqrt(1.0 - beta2 ** self.__t) / (1.0 - beta1 ** self.__t))
        
        return grad_W, grad_b
        
    def __l2(self, l, grad_W):
        grad_W += l * self.W_
        self.W_ -= self.__eta * grad_W
            
    def __l1(self, l, grad_W):
        grad_W += l * np.sign(self.W_)
        self.W_ -= self.__eta * grad_W
        
    def __compute_v(self, grad_W, grad_b, beta1):
        self.__v_w = (beta1 * self.__v_w + (1.0 - beta1) * grad_W) 
        self.__v_b = (beta1 * self.__v_b + (1.0 - beta1) * grad_b)
    
    def __compute_s(self, grad_W, grad_b, beta2):
        self.__s_w = (beta2 * self.__s_w + (1.0 - beta2) * grad_W ** 2) / (1.0 - beta2 ** self.__t)
        self.__s_b = (beta2 * self.__s_b + (1.0 - beta2) * grad_b ** 2) / (1.0 - beta2 ** self.__t)
                    
    @property
    def delta(self):
        return self.__delta
    
    @property
    def output(self):
        return self.__output
    
    
class DNN:
    
    cost_functions = {'logistic': (logistic_cost, logistic_grad), 'mse': (mse, mse_grad), 
                      'cross_entropy': (categorical_cross_entropy, grad_categorical_cross_entropy)}
    
    REGULARIZATIONS = {'l2', 'l1', 'dropout'}
    ALGORITHMS = {'mini_batch', 'adam', 'momentum', 'rmsprop'}
    
    def __init__(self, n_iter, eta, cost_function, batch_size, 
                 learning_schedule=None, algorithm='mini_batch', regularization=None,  
                 cost_ratio=10, shuffle=False, print_cost=True, **kwargs):
        
        if regularization is not None:
            assert regularization in self.REGULARIZATIONS, f'{regularization} not in {self.REGULARIZATIONS}'

        assert algorithm in self.ALGORITHMS, f'{algorithm} not in {self.ALGORITHMS}'
        
        self.print_cost = print_cost
         
        self.n_iter = n_iter
        self.eta = eta
        self.batch_size = batch_size
        self.learning_schedule = learning_schedule
        self.cost_function, self.cost_grad = self.cost_functions[cost_function]
        
        self.beta1 = None
        self.beta2 = None
        self.keep_prob = None
        self.l = None
        self.regularization_info = None
        
        if algorithm == 'momentum':
            self.beta1 = kwargs['beta1']
            self.beta2 = 0
            
        elif algorithm == 'adam':
            self.beta1 = kwargs['beta1']
            self.beta2 = kwargs['beta2']
            
        elif algorithm == 'rmsprop':
            self.beta2 = kwargs['beta2']
            self.beta1 = 0
            
        if regularization in ['l2', 'l1']:
            self.l = kwargs['Lambda'] 
            self.regularization_info = (regularization, self.l)
            
        elif regularization == 'dropout':
            self.keep_prob = kwargs['keep_prob']
            self.regularization_info = (regularization, self.keep_prob)
            
        self.shuffle = shuffle
        
        self.algorithm_info = (algorithm, self.beta1, self.beta2) if algorithm != 'mini_batch' else algorithm

        self.layers = []
        self.__current_nodes = None
        self.fitted = False
        self.cost_ratio = cost_ratio
        
    def fit(self, X, y):
        """
        X: array. shape = [n_features, n_samples]
            - Training dataset
        y: array. shape = [n_classes, n_samples]
            - Target values
        
        This function computes the algorithm itself so we can make predictions. 
        """
        if self.batch_size is None: 
            self.batch_size = X.shape[1]
                
        assert len(self.layers) > 0, 'Add a Layer object to implement a feedforward neural network'    
        assert self.batch_size <= X.shape[1], 'The batch size can not be greater than the number of samples'
        assert y.ndim > 1, 'The shape of y must be (n_classes, n_samples)'
        
        self.cost_ = []
        eta0 = self.eta
        
        for n in range(self.n_iter):
            
            batch_index = np.arange(0, X.shape[1])
            
            if self.shuffle:
                assert self.batch_size is not None, 'Set the batch_size argument'
                shuffle_index = np.random.choice(range(X.shape[1]), size=X.shape[1], replace=False)
                X = X[:, shuffle_index]
                y = y[:, shuffle_index]
                batch_index = np.random.choice(range(X.shape[1]), size=self.batch_size, replace=False)
    
            x_batch = X[:, batch_index]
            y_batch = y[:, batch_index]
            
            self.__forward(x_batch)
            gradient = self.__cost_gradient(y_batch)
            self.__backward(gradient)
            self.__update_weights(gradient, epoch=n+1)
            
            cost = self.compute_cost(y_batch)
            
            if not (n + 1) % self.cost_ratio and self.print_cost:
                self.cost_.append(cost)
                print(f'Cost {n + 1}th epoch:, ', np.round(cost, decimals=3))
                print('-' * 20)
                
            self.eta = self.learning_schedule(n, eta0) if self.learning_schedule is not None else self.eta
            
        self.fitted = True
            
        return self
               
    def add_layer(self, layer, n_features=None):
        """
        layer: Layer
        n_features: int
            - It's optional when there are layers in the network. Otherwise insert it. 
        
        We add a Layer to the Network. 
        """
        
        if len(self.layers) > 0:
            assert n_features is None, 'Do not insert n_features if len(layers) > 0'
        else:
            assert n_features is not None, 'Insert the number of features of the dataset'
        
        if self.__current_nodes is None:
            layer.random_init(n_features)
        else:
            layer.random_init(self.__current_nodes)
        
        self.__current_nodes = layer.nodes
        self.layers.append(layer)
        
    def compute_cost(self, y):
        """
        y: array. shape = [n_samples, n_classes]
        Compute the cost 
        """
        assert self.__last_activation is not None, 'Compute the forward step before implementing the cost'
        cost = self.cost_function(y, self.__last_activation)
        return np.sum(cost) / y.shape[0]
    
    def __forward(self, X, test=False):
        """
        Compute the forward propagation step
        """
        current_input = X
        self.__last_activation = None
        self.__last_layer = None
        
        layers = []
        self.__random_dropouts = {}
        
        for layer in self.layers:
            layer.forward(current_input)
            current_input = layer.output         
            layers.append(layer)
        
        self.layers = layers.copy()
        self.__last_activation = current_input
        self.__last_layer = self.layers[-1]
        
    def __backward(self, gradient=None):
        """
        Backpropagation step
        """
        assert self.__last_activation is not None, 'Apply forward propagation before implementing backward'
        assert gradient is not None, 'Insert the gradient of the cost function'
        
        layers = []
        last_Z = self.__last_layer.cache_['Z']
        forward_delta = gradient * self.__last_layer.deriv_activation(last_Z)
        forward_weights = self.__last_layer.cache_['W']
                
        # we do not need to compute the error of the last layer since we already have it: forward_delta
        for layer in reversed(self.layers[:-1]):
            layer.backward(forward_delta, forward_weights)
            forward_delta = layer.delta
            forward_weights = layer.cache_['W']
            layers.append(layer)
            
        self.layers = layers[::-1].copy()
        self.layers.append(self.__last_layer)
        
    def __update_weights(self, gradient=None, epoch=None):
        """
        Gradient descent. 
        """
        layers = []
        for l, layer in enumerate(self.layers):
            if l == len(self.layers) - 1:
                layer.update_weights(self.eta, last_layer=True, grad=gradient, t=epoch,  
                                     algorithm=self.algorithm_info, 
                                     regularization=self.regularization_info)
                                     
            layer.update_weights(self.eta, grad=gradient, t=epoch, 
                                 algorithm=self.algorithm_info, 
                                 regularization=self.regularization_info)
            layers.append(layer)
            
        self.layers = layers.copy()
        
    def __cost_gradient(self, y):
        return self.cost_grad(y, self.__last_activation)
    
    def predict(self, X, softmax=False):
        """
        X: array. shape = [n_features, n_samples]
        """
        assert self.fitted, 'Fit the model to the data in order to predict'
        self.__forward(X, test=True)
        if softmax:
            y_pred = np.exp(self.__last_activation) / (np.sum(np.exp(self.__last_activation), axis=0))
            return np.argmax(y_pred, axis=0).squeeze(), y_pred

        y_pred = np.where(self.__last_activation >= 0.5, 1, 0)
        return np.argmax(y_pred, axis=0).squeeze()
        
        
