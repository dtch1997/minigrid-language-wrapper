import numpy as np
from copy import deepcopy
from gymnasium import spaces
from gymnasium.core import ObservationWrapper, ActionWrapper

import minigrid
from minigrid.core.actions import Actions
from minigrid.core.constants import (
    IDX_TO_OBJECT,
    IDX_TO_COLOR,
    STATE_TO_IDX,
    OBJECT_TO_IDX,
    COLOR_TO_IDX,
)

IDX_TO_STATE = {v: k for k, v in STATE_TO_IDX.items()}


class RoomgridObservationWrapper(ObservationWrapper):

    """Replace the 'image' observation with an observation of the current room"""

    def __init__(self, env):
        super().__init__(env)

        assert isinstance(
            self.unwrapped, minigrid.roomgrid.RoomGrid
        ), "This wrapper only works with RoomGrid environments"
        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.env.room_size, self.env.room_size, 3),  # number of cells
            dtype="uint8",
        )

        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, observation) -> dict:
        env = self.unwrapped

        room = env.room_from_pos(*env.agent_pos)
        agentX, agentY = env.agent_pos
        topX, topY = room.top
        room_grid = env.grid.slice(topX, topY, env.room_size, env.room_size)

        # Calculate the agent's position in the room
        agentXRoom, agentYRoom = agentX - topX, agentY - topY
        room_grid_enc = room_grid.encode()

        # Replace the agent's position in the room with the agent
        # Notes:
        # - This overwrites any other object in the agent's position
        # E.g. if the agent is standing on a door, the door will be replaced with the agent
        # - The agent's direction is not encoded in the room grid
        room_grid_enc[agentXRoom][agentYRoom] = np.array(
            [OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], env.agent_dir]
        )

        return {**observation, "image": room_grid_enc}
