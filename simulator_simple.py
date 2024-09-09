import pybullet as p
import pybullet_data
import numpy as np
import time
from hexapod.simulator import Simulator
from hexapod.controllers.kinematic import Controller

# Define a function to run the simulation using the provided controller and gait
def run_simple_simulation(duration=5.0, visualiser=True):

    # Initialize the Controller with the selected gait
    controller = Controller()

    # Initialize the simulator and pass the controller to it
    simulator = Simulator(controller, visualiser=visualiser, follow=True, failed_legs=[])

    # Run the simulation for the defined duration (e.g., 5 seconds)
    for t in np.arange(0, duration, step=simulator.dt):
        simulator.step()

    # Terminate the simulation
    simulator.terminate()

if __name__ == "__main__":
    # Run the simulation for 5 seconds with visualization enabled
    run_simple_simulation(duration=5.0, visualiser=True)
