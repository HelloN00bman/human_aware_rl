from dependencies import *
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, PlayerState, ObjectState, OvercookedState
from overcooked_ai_py.planning.planners import MediumLevelPlanner
from overcooked_ai_py.mdp.actions import Action, Direction
import pdb
from state_featurization_for_irl import *


from itertools import product

import numpy as np
import numpy.random as rn





def test_behavior_cloning():
    X, Y = run_featurization_bc()

    print("X shaoe", X.shape)
    print('Y', Y.shape)

    device = torch.device('cpu')
    # device = torch.device('cuda') # Uncomment this to run on GPU

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 256, 107, 100, 34

    # Create random Tensors to hold inputs and outputs
    x = torch.Tensor(X)
    y = torch.Tensor(Y)

    # Use the nn package to define our model as a sequence of layers. nn.Sequential
    # is a Module which contains other Modules, and applies them in sequence to
    # produce its output. Each Linear Module computes output from input using a
    # linear function, and holds internal Tensors for its weight and bias.
    # After constructing the model we use the .to() method to move it to the
    # desired device.
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
    ).to(device)

    # The nn package also contains definitions of popular loss functions; in this
    # case we will use Mean Squared Error (MSE) as our loss function. Setting
    # reduction='sum' means that we are computing the *sum* of squared errors rather
    # than the mean; this is for consistency with the examples above where we
    # manually compute the loss, but in practice it is more common to use mean
    # squared error as a loss by setting reduction='elementwise_mean'.
    loss_fn = torch.nn.MSELoss(reduction='sum')

    learning_rate = 1e-4
    n_epochs = 20000
    for t in range(n_epochs):
        # Forward pass: compute predicted y by passing x to the model. Module objects
        # override the __call__ operator so you can call them like functions. When
        # doing so you pass a Tensor of input data to the Module and it produces
        # a Tensor of output data.
        y_pred = model(x)

        # Compute and print loss. We pass Tensors containing the predicted and true
        # values of y, and the loss function returns a Tensor containing the loss.
        # y_pred = np.argmax(y_pred, axis=0)
        # print('y_pred, y', (y_pred, y))

        loss = loss_fn(y_pred, y)
        # print(t, loss.item())

        # Zero the gradients before running the backward pass.
        model.zero_grad()

        # Backward pass: compute gradient of the loss with respect to all the learnable
        # parameters of the model. Internally, the parameters of each Module are stored
        # in Tensors with requires_grad=True, so this call will compute gradients for
        # all learnable parameters in the model.
        loss.backward()

        # Update the weights using gradient descent. Each parameter is a Tensor, so
        # we can access its data and gradients like we did before.
        with torch.no_grad():
            for param in model.parameters():
                param.data -= learning_rate * param.grad


    # TEST
    x = torch.Tensor(X)
    y = torch.Tensor(Y)
    y_pred = model(x).detach().numpy()
    y_true = Y
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_true, axis=1)

    accuracy = 0
    for i in range(len(y_pred)):
        if y_true[i] == y_pred[i]:
            accuracy += 1

    print("ACCURACY", accuracy/len(y_pred))





if __name__ == '__main__':
    test_behavior_cloning()












