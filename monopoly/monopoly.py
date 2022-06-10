import gym
from gym import spaces
import numpy as np
import random

num_position = 40
num_jail_turn = 3
num_GOOJ_card = 2

'''
property levels:
- 0: mortgaged
- 1: 0 house
- 2: 1 house
- 3: 2 houses
- 4: 3 houses
- 5: 4 houses
- 6: hotel
'''
property_levels = 7
property_positions = (
    0, 1, 0, 1, 0, 1, 1, 0, 1, 1,
    0, 1, 1, 1, 1, 1, 1, 0, 1, 1,
    0, 1, 0, 1, 1, 1, 1, 1, 1, 1,
    0, 1, 1, 0, 1, 1, 0, 1, 0, 1,
)
property_positions_upgradeable = (
    0, 1, 0, 1, 0, 0, 1, 0, 1, 1,
    0, 1, 0, 1, 1, 0, 1, 0, 1, 1,
    0, 1, 0, 1, 1, 0, 1, 1, 0, 1,
    0, 1, 1, 0, 1, 0, 0, 1, 0, 1,
)
property_cost = (
    0,  40,   0,  60,   0, 200, 100,   0, 100, 120,
    0, 140, 150, 140, 160, 200, 180,   0, 180, 200, 
    0, 220,   0, 220, 240, 200, 260, 260, 150, 280,
    0, 300, 300,   0, 320, 200,   0, 350,   0, 400,
)
property_mortgage_return = tuple(c // 2 for c in property_cost)
property_unmortgage_cost = tuple(int(c * 1.1) for c in property_mortgage_return) 
zone_upgrade_cost = (
    [50, 60, 65, 65, 65],
    [100, 120, 130, 130, 130],
    [150, 180, 195, 195, 195],
    [200, 240, 260, 260, 260],
)
# property_upgrade_cost[position][current_level]
property_upgrade_cost = tuple(
    tuple(
        [property_unmortgage_cost[i]] + (
            zone_upgrade_cost[i // 10]
            if upgradeable
            else [0] * 5
        ) + [0]
    ) for i, upgradeable in enumerate(property_positions_upgradeable)
)
# property_downgrade_return[position][current_level]
property_downgrade_return = tuple(
    tuple(
        [0] + [property_mortgage_return[i]] + 5 * [
            zone_upgrade_cost[i // 10][0] if upgradeable else 0
        ]
    ) for i, upgradeable in enumerate(property_positions_upgradeable)
)
property_rent = (
    tuple([0] * 7), (0, 2, 10, 30, 90, 160, 250), tuple([0] * 7), (0, 4, 20, 60, 180, 320, 450), tuple([0] * 7),
    (0, 25, 50, 100, 200, 0, 0), (0, 6, 30, 90, 270, 400, 550), tuple([0] * 7), (0, 6, 30, 90, 270, 400, 550), (0, 8, 40, 100, 300, 450, 600),
    tuple([0] * 7), (0, 10, 50, 150, 450, 625, 750), (0, 4, 10, 0, 0, 0, 0), (0, 10, 50, 150, 450, 625, 750), (0, 12, 60, 180, 500, 700, 900),
    (0, 25, 50, 100, 200, 0, 0), (0, 14, 70, 200, 550, 750, 950), tuple([0] * 7), (0, 14, 70, 200, 550, 750, 950), (0, 16, 80, 220, 600, 800, 1000),
    tuple([0] * 7), (0, 18, 90, 250, 700, 875, 1050), tuple([0] * 7), (0, 18, 90, 250, 700, 875, 1050), (0, 20, 100, 300, 750, 925, 1100),
    (0, 25, 50, 100, 200, 0, 0), (0, 22, 110, 330, 800, 975, 1150), (0, 22, 110, 330, 800, 975, 1150), (0, 4, 10, 0, 0, 0, 0), (0, 24, 120, 360, 850, 1025, 1200),
    tuple([0] * 7), (0, 26, 130, 390, 900, 1100, 1275), (0, 26, 130, 390, 900, 1100, 1275), tuple([0] * 7),  (0, 28, 150, 450, 1000, 1200, 1400),
    (0, 25, 50, 100, 200, 0, 0), tuple([0] * 7), (0, 35, 175, 500, 1100, 1300, 1500), (0, 50, 200, 600, 1400, 1700, 2000),
) #TODO

class Monopoly(gym.Env):
    metadata = None
    
    def __init__(self, num_players=5):
        self.num_players = num_players
        self.observation_space = spaces.Dict({
            'turn': spaces.Discrete(num_players),
            'position': spaces.MultiDiscrete([num_position] * num_players),
            'jail': spaces.MultiBinary(num_players),
            'jailTurn': spaces.MultiDiscrete([num_jail_turn] * num_players),
            'cash': spaces.Box(shape=(num_players,), dtype=int),
            'bankrupt': spaces.MultiBinary(num_players),
            'GOOJ': spaces.MultiDiscrete([num_GOOJ_card+1] * num_players),
            'propertyOwner': spaces.Box(low=-1, high=num_players-1, shape=(num_position,), dtype=int),
            'propertyUpgrade': spaces.MultiDiscrete([property_levels] * num_position),
            'diceRolled': spaces.Discrete(2),
            'auctionMode': spaces.Discrete(2),
            'auctionProperty': spaces.Discrete(num_position),
            'auctioner': spaces.Discrete(num_players),
            'auctionValue': spaces.Box(low=0, shape=(1,), dtype=int),
        })
        self.action_space = spaces.Dict({
            # pre dice roll
            'GOOJ': spaces.Discrete(2),
            # pre/post dice roll
            'auctionProperty': spaces.Discrete(num_position),
            'auctionValue': spaces.Box(low=0, shape=(1,), dtype=int),
            'propertyChange': spaces.Box(low=-6, high=6, shape=(num_position,), dtype=int),
            # auction mode
            'bid': spaces.Discrete(2)
        })

    def _roll_dice(self):
        a, b = self.random.randint(1, 6), self.random.randint(1, 6)
        return a + b, a == b

    def _get_obs():
        return None

    def _get_info():
        return None

    def reset(self, seed=None, return_info=False, options=None):
        super().reset(seed=seed)
        self.random = random.Random(seed)
        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation
    
    def step(self, action):
        observation = self._get_obs()
        reward = 0
        done = False
        info = self._get_info()
        return observation, reward, done, info