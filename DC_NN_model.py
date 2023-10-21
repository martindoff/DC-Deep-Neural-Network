""" Difference of Convex deep Neural Network (DC-NN) model - coupled tank

Approximation of the coupled tank dynamics by a deep NN model with DC structure.

(c) Martin Doff-Sotta, University of Oxford (martin.doff-sotta@eng.ox.ac.uk)

"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.constraints import NonNeg
import matplotlib.pyplot as plt
import param_init as param
from tank_model import f
from mpl_toolkits import mplot3d

# Deep NN with convex structure 
def convex_NN(N_layer, N_node):
    """ Create a densely connected feedforward neural network with convex input-output map
    Input: 
        - N_layer: number of hidden layers
        - N_node: number of units per layer
    Output: neural network model
    """
    
    # Input
    input = keras.Input(shape=(3,))
    x = input
    x = layers.Dense(N_node)(input)
    x = layers.ReLU()(x)
    
    # Add densely connected layers (with nonnegatively constrained kernel weights)
    for i in range(N_layer):
        x1 = layers.Dense(N_node, kernel_constraint=NonNeg())(x) # from current activation
        x2 = layers.Dense(N_node)(input)                         # from network input
        x = layers.Add()([x1, x2])
        x = layers.ReLU()(x)                                     # compute next activation
    
    # Output layers (with nonnegatively constrained kernel weights)
    output = layers.Dense(2, kernel_constraint=NonNeg())(x)
    
    return keras.Model(input, output)

# main
if __name__ == "__main__":
    """ 
    Test the DC neural network architecture on an example
    
    """
    
    load = False # set to False if model has to be retrained
    
    # Generate training points:
    N_train = 100000
    N_state = 2
    N_input = 1
    x_train = (param.x_max[0]-param.x_min[0])*np.random.rand(N_state, 
                                                                 N_train) + param.x_min[0]
    u_train = (param.u_max[0]-param.u_min[0])*np.random.rand(N_input, 
                                                                 N_train) + param.u_min[0]
    y_train = np.empty_like(x_train)
    for i in range(N_train):
        y_train[:, i] = f(x_train[:, i], u_train[:, i], param)
    

    # Generate test data
    N_test = 10
    x_test = (param.x_max[0]-param.x_min[0])*np.random.rand(N_state, 
                                                                  N_test) + param.x_min[0]
    u_test = (param.u_max[0]-param.u_min[0])*np.random.rand(N_state, 
                                                                  N_test) + param.u_min[0]
    y_test = np.empty_like(x_test)
    for i in range(N_test):
        y_test[:, i] = f(x_test[:, i], u_test[:, i], param)
        
    
    
    # Build model
    input = keras.Input(shape=(N_state+N_input,))
    
    model_f1 = convex_NN(1, 64)  # input-output convex NN
    model_f2 = convex_NN(1, 64)  # input-output convex NN
    
    f1 = model_f1(input)
    f2 = model_f2(input)
    
    output = layers.Subtract()([f1, f2])  # Difference of Convex Neural Network (DC-NN)
    
    f_DC = keras.Model(inputs=input, outputs=output)

    # Compile 
    f_DC.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    
    
    # Load or train model
    if load:  # load existing model
    
        # Restore the weights
        f_DC.load_weights('./model/f_DC').expect_partial()

    else:  # train new model
        
        # Train model
        history = f_DC.fit(np.vstack([x_train, u_train]).T, y_train.T, batch_size=64, 
                       epochs=10, validation_split=0.2)
        
        # Save the weights
        f_DC.save_weights('./model/f_DC')
    
        # Training and validation loss
        loss_train = history.history['loss']
        loss_val = history.history['val_loss']
        epochs = range(1, len(loss_train)+1)
    
        fig, ax = plt.subplots()
        ax.plot(epochs, loss_train, 'bo', label='Training loss')
        ax.plot(epochs, loss_val, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        ax.set(xlim=(epochs[0], epochs[-1]), ylim=(0, 0.2))
        ax.legend()
    
        #plt.savefig('plot/loss.eps', format='eps')
        plt.show()
    
    # Plot results
    V = np.linspace(param.u_min, param.u_max, 10)             # voltage
    h1 = np.linspace(param.x_min[0], param.x_max[0], N_test)  # height tank 1
    h2 = np.linspace(param.x_min[1], param.x_max[1], N_test)  # height tank 2

    H1, H2 = np.meshgrid(h1, h2)                              
    for volt in V:  # projection (fixed voltage input)
        
        # Initialisation
        F = np.zeros((N_test**2, N_state))           # dynamic model f (reference)
        Y = np.zeros((N_test, N_test, N_state))      # NN prediction for f in f = f1 - f2
        Y1 = np.zeros((N_test, N_test, N_state))     # NN prediction for f1 in f = f1 - f2
        Y2 = np.zeros((N_test, N_test, N_state))     # NN prediction for f2 in f = f1 - f2
        x = np.zeros((N_test**2, N_state+N_input))
        k = 0
        
        # Assemble a vector of size (num_instances,features) for fast evaluation in keras
        for height1 in h1:
            for height2 in h2: 
                x[k, :] = np.array([height1, height2, volt], dtype=object)
                F[k, :] = f(np.array([height1, height2], dtype=object), volt, param) 
                k += 1
        
        # Model predictions (fast evaluation on x)
        y = f_DC.predict(x)       # predict f  DC s.t. f = f1 - f2 (f1, f2 convex)
        y1 = model_f1.predict(x)  # predict f1 convex s.t. f = f1 - f2 (f1, f2 convex)
        y2 = model_f2.predict(x)  # predict f2 convex s.t. f = f1 - f2 (f1, f2 convex)
               
        # Arrange data in grid format for 3D plot
        k = 0
        for i in range(N_test):
            for j in range(N_test):                        
                Y[j, i, :] = y[k]
                Y1[j, i, :] = y1[k] 
                Y2[j, i, :] = y2[k]
                k += 1
        
        # Plots       
        fig = plt.figure(figsize=plt.figaspect(0.5))
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        
        ax.scatter(x[:,0], x[:,1], F[:, 0], label='tank model (ref)')
        c0 = ax.plot_surface(H1, H2, Y[:,:,0], alpha=0.7, linewidth=0, antialiased=True, 
                             shade=True, label='DCNN: $f=f_{1}-f_{2}$')
        c1 = ax.plot_surface(H1, H2, Y1[:,:,0], alpha=0.7, linewidth=0, antialiased=True, 
                             shade=True, label='DCNN: $f_{1}$')
        c2 = ax.plot_surface(H1, H2, Y2[:,:,0], alpha=0.7, linewidth=0, antialiased=True, 
                             shade=True, label='DCNN: $f_{2}$')

        c0._facecolors2d = c0._facecolor3d
        c0_edgecolors2d = c0._edgecolor3d
        c1._facecolors2d = c1._facecolor3d
        c1._edgecolors2d = c1._edgecolor3d
        c2._facecolors2d = c2._facecolor3d
        c2._edgecolors2d = c2._edgecolor3d
            
        plt.title('DC decomposition, u = {} V'.format(round(volt[0], 1)))
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$\dot{x}_1$')
        ax.legend()
        
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        
        ax.scatter(x[:,0], x[:,1], F[:, 1], label='tank model (ref)')
        c0 = ax.plot_surface(H1, H2, Y[:,:,1], alpha=0.7, linewidth=0, antialiased=True, 
                             shade=True, label='DCNN: $f=f_{1}-f_{2}$')
        c1 = ax.plot_surface(H1, H2, Y1[:,:,1], alpha=0.7, linewidth=0, antialiased=True, 
                             shade=True, label='DCNN: $f_{1}$')
        c2 = ax.plot_surface(H1, H2, Y2[:,:,1], alpha=0.7, linewidth=0, antialiased=True, 
                             shade=True, label='DCNN: $f_{2}$')

        c0._facecolors2d = c0._facecolor3d
        c0_edgecolors2d = c0._edgecolor3d
        c1._facecolors2d = c1._facecolor3d
        c1._edgecolors2d = c1._edgecolor3d
        c2._facecolors2d = c2._facecolor3d
        c2._edgecolors2d = c2._edgecolor3d

        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$\dot{x}_2$')
        ax.legend()
          
    plt.show()
    
    #uncomment to plot curves (this will take more time and generate a lot of plots)
    """V = [0, 1, 3, 5, 7, 10, 12, 15, 18, 20, 24]
    Height = [0.1, 1, 3, 5, 7, 10, 12, 14, 15]
    for volt in V:
        for height in Height:
            h1 = np.linspace(0, 15, N_test)
            h2 = height*np.ones_like(h1)
            u = volt*np.ones_like(h1)
            y = np.empty_like(y_test)
            for i in range(N_test):
                y[:, i, None] = f(np.vstack([h1[i], h2[i]]), u[i, None], param)
    
            plt.figure()
            plt.subplot(211)
            plt.plot(h1, y[0, :], '--b', label='ref')
            plt.plot(h1, f_DC.predict(np.vstack([h1, h2, u]).T)[:, 0],
                     '-r', label='$f=f_1-f_2$')
            plt.plot(h1, model_f1.predict(np.vstack([h1, h2, u]).T)[:, 0], 
                     '-b', label='$f_1$')
            plt.plot(h1, model_f2.predict(np.vstack([h1, h2, u]).T)[:, 0], 
                     '-g', label='$f_2$')
            plt.title('$h_2$ = {} cm, u = {} V'.format(height, volt))
            plt.xlabel('$h_1$')
            plt.ylabel('$\dot{h}_1$')
            plt.legend()
    
            plt.subplot(212)
            plt.plot(h1, y[1, :], '--b', label='ref')
            plt.plot(h1, f_DC.predict(np.vstack([h1, h2, u]).T)[:, 1], 
                     '-r', label='$f=f_1-f_2$')
            plt.plot(h1, model_f1.predict(np.vstack([h1, h2, u]).T)[:, 1], 
                     '-b', label='$f_1$')
            plt.plot(h1, model_f2.predict(np.vstack([h1, h2, u]).T)[:, 1], 
                     '-g', label='$f_2$')
            plt.xlabel('$h_1$')
            plt.ylabel('$\dot{h}_2$')
            plt.legend()
        
    
    plt.show()"""
    
    # Graph
    keras.utils.plot_model(f_DC, "f.png", show_shapes=True)
    keras.utils.plot_model(model_f1, "f1.png", show_shapes=True)
    keras.utils.plot_model(model_f2, "f2.png", show_shapes=True)
    
    # uncomment to display weights
    """print("Weights: ")
    for w in model_f1.get_weights():
        print("new w: ")
        print(w)"""  