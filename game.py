import json
from multiprocessing.managers import BaseListProxy
import os
import pickle
import random
from abc import ABC, abstractmethod
from queue import Empty, Full, LifoQueue, Queue
from threading import Lock, Thread
from time import time

import ray
from utils import DOCKER_VOLUME, create_dirs

from human_aware_rl.rllib.rllib import load_agent
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import (
    NO_COUNTERS_PARAMS,
    MotionPlanner,
)

# Relative path to where all static pre-trained agents are stored on server
AGENT_DIR = None

# Maximum allowable game time (in seconds)
MAX_GAME_TIME = None


map_Name = ""
myId = 0
use200ScoreStopSoup = True

def _configure(max_game_time, agent_dir):
    global AGENT_DIR, MAX_GAME_TIME
    MAX_GAME_TIME = max_game_time
    AGENT_DIR = agent_dir


def fix_bc_path(path):
    """
    Loading a PPO agent trained with a BC agent requires loading the BC model as well when restoring the trainer, even though the BC model is not used in game
    For now the solution is to include the saved BC model and fix the relative path to the model in the config.pkl file
    """

    import dill

    # the path is the agents/Rllib.*/agent directory
    agent_path = os.path.dirname(path)
    with open(os.path.join(agent_path, "config.pkl"), "rb") as f:
        data = dill.load(f)
    bc_model_dir = data["bc_params"]["bc_config"]["model_dir"]
    last_dir = os.path.basename(bc_model_dir)
    bc_model_dir = os.path.join(agent_path, "bc_params", last_dir)
    data["bc_params"]["bc_config"]["model_dir"] = bc_model_dir
    with open(os.path.join(agent_path, "config.pkl"), "wb") as f:
        dill.dump(data, f)


class Game(ABC):

    """
    Class representing a game object. Coordinates the simultaneous actions of arbitrary
    number of players. Override this base class in order to use.

    Players can post actions to a `pending_actions` queue, and driver code can call `tick` to apply these actions.


    It should be noted that most operations in this class are not on their own thread safe. Thus, client code should
    acquire `self.lock` before making any modifications to the instance.

    One important exception to the above rule is `enqueue_actions` which is thread safe out of the box
    """

    # Possible TODO: create a static list of IDs used by the class so far to verify id uniqueness
    # This would need to be serialized, however, which might cause too great a performance hit to
    # be worth it

    EMPTY = "EMPTY"

    class Status:
        DONE = "done"
        ACTIVE = "active"
        RESET = "reset"
        INACTIVE = "inactive"
        ERROR = "error"

    def __init__(self, *args, **kwargs):
        """
        players (list): List of IDs of players currently in the game
        spectators (set): Collection of IDs of players that are not allowed to enqueue actions but are currently watching the game
        id (int):   Unique identifier for this game
        pending_actions List[(Queue)]: Buffer of (player_id, action) pairs have submitted that haven't been commited yet
        lock (Lock):    Used to serialize updates to the game state
        is_active(bool): Whether the game is currently being played or not
        """
        self.players = []
        self.spectators = set()
        self.pending_actions = []
        self.id = kwargs.get("id", id(self))
        self.lock = Lock()
        self._is_active = False

    @abstractmethod
    def is_full(self):
        """
        Returns whether there is room for additional players to join or not
        """
        pass

    @abstractmethod
    def apply_action(self, player_idx, action):
        """
        Updates the game state by applying a single (player_idx, action) tuple. Subclasses should try to override this method
        if possible
        """
        pass

    @abstractmethod
    def is_finished(self):
        """
        Returns whether the game has concluded or not
        """
        pass

    def is_ready(self):
        """
        Returns whether the game can be started. Defaults to having enough players
        """
        return self.is_full()

    @property
    def is_active(self):
        """
        Whether the game is currently being played
        """
        return self._is_active

    @property
    def reset_timeout(self):
        """
        Number of milliseconds to pause game on reset
        """
        return 3000

    def apply_actions(self):
        """
        Updates the game state by applying each of the pending actions in the buffer. Is called by the tick method. Subclasses
        should override this method if joint actions are necessary. If actions can be serialized, overriding `apply_action` is
        preferred
        """
        for i in range(len(self.players)):
            try:
                while True:
                    action = self.pending_actions[i].get(block=False)
                    self.apply_action(i, action)
            except Empty:
                pass

    def activate(self):
        """
        Activates the game to let server know real-time updates should start. Provides little functionality but useful as
        a check for debugging
        """
        self._is_active = True

    def deactivate(self):
        """
        Deactives the game such that subsequent calls to `tick` will be no-ops. Used to handle case where game ends but
        there is still a buffer of client pings to handle
        """
        self._is_active = False

    def reset(self):
        """
        Restarts the game while keeping all active players by resetting game stats and temporarily disabling `tick`
        """
        if not self.is_active:
            raise ValueError("Inactive Games cannot be reset")
        if self.is_finished():
            return self.Status.DONE
        self.deactivate()
        self.activate()
        return self.Status.RESET

    def needs_reset(self):
        """
        Returns whether the game should be reset on the next call to `tick`
        """
        return False

    def tick(self):
        """
        Updates the game state by applying each of the pending actions. This is done so that players cannot directly modify
        the game state, offering an additional level of safety and thread security.

        One can think of "enqueue_action" like calling "git add" and "tick" like calling "git commit"

        Subclasses should try to override `apply_actions` if possible. Only override this method if necessary
        """
        if not self.is_active:
            return self.Status.INACTIVE
        if self.needs_reset():
            self.reset()
            return self.Status.RESET

        self.apply_actions()
        return self.Status.DONE if self.is_finished() else self.Status.ACTIVE

    def enqueue_action(self, player_id, action):
        """
        Add (player_id, action) pair to the pending action queue, without modifying underlying game state

        Note: This function IS thread safe
        """
        if not self.is_active:
            # Could run into issues with is_active not being thread safe
            return
        if player_id not in self.players:
            # Only players actively in game are allowed to enqueue actions
            return
        try:
            player_idx = self.players.index(player_id)
            self.pending_actions[player_idx].put(action)
        except Full:
            pass

    def get_state(self):
        """
        Return a JSON compatible serialized state of the game. Note that this should be as minimalistic as possible
        as the size of the game state will be the most important factor in game performance. This is sent to the client
        every frame update.
        """
        return {"players": self.players}

    def to_json(self):
        """
        Return a JSON compatible serialized state of the game. Contains all information about the game, does not need to
        be minimalistic. This is sent to the client only once, upon game creation
        """
        return self.get_state()

    def is_empty(self):
        """
        Return whether it is safe to garbage collect this game instance
        """
        return not self.num_players

    def add_player(self, player_id, idx=None, buff_size=-1):
        """
        Add player_id to the game
        """
        if self.is_full():
            raise ValueError("Cannot add players to full game")
        if self.is_active:
            raise ValueError("Cannot add players to active games")
        if not idx and self.EMPTY in self.players:
            idx = self.players.index(self.EMPTY)
        elif not idx:
            idx = len(self.players)

        padding = max(0, idx - len(self.players) + 1)
        for _ in range(padding):
            self.players.append(self.EMPTY)
            self.pending_actions.append(self.EMPTY)

        self.players[idx] = player_id
        self.pending_actions[idx] = Queue(maxsize=buff_size)

    def add_spectator(self, spectator_id):
        """
        Add spectator_id to list of spectators for this game
        """
        if spectator_id in self.players:
            raise ValueError("Cannot spectate and play at same time")
        self.spectators.add(spectator_id)

    def remove_player(self, player_id):
        """
        Remove player_id from the game
        """
        try:
            idx = self.players.index(player_id)
            self.players[idx] = self.EMPTY
            self.pending_actions[idx] = self.EMPTY
        except ValueError:
            return False
        else:
            return True

    def remove_spectator(self, spectator_id):
        """
        Removes spectator_id if they are in list of spectators. Returns True if spectator successfully removed, False otherwise
        """
        try:
            self.spectators.remove(spectator_id)
        except ValueError:
            return False
        else:
            return True

    def clear_pending_actions(self):
        """
        Remove all queued actions for all players
        """
        for i, player in enumerate(self.players):
            if player != self.EMPTY:
                queue = self.pending_actions[i]
                queue.queue.clear()

    @property
    def num_players(self):
        return len([player for player in self.players if player != self.EMPTY])

    def get_data(self):
        """
        Return any game metadata to server driver.
        """
        return {}


class DummyGame(Game):

    """
    Standin class used to test basic server logic
    """

    def __init__(self, **kwargs):
        super(DummyGame, self).__init__(**kwargs)
        self.counter = 0

    def is_full(self):
        return self.num_players == 2

    def apply_action(self, idx, action):
        pass

    def apply_actions(self):
        self.counter += 1

    def is_finished(self):
        return self.counter >= 100

    def get_state(self):
        state = super(DummyGame, self).get_state()
        state["count"] = self.counter
        return state


class DummyInteractiveGame(Game):

    """
    Standing class used to test interactive components of the server logic
    """

    def __init__(self, **kwargs):
        super(DummyInteractiveGame, self).__init__(**kwargs)
        self.max_players = int(
            kwargs.get("playerZero", "human") == "human"
        ) + int(kwargs.get("playerOne", "human") == "human")
        self.max_count = kwargs.get("max_count", 30)
        self.counter = 0
        self.counts = [0] * self.max_players

    def is_full(self):
        return self.num_players == self.max_players

    def is_finished(self):
        return max(self.counts) >= self.max_count

    def apply_action(self, player_idx, action):
        if action.upper() == Direction.NORTH:
            self.counts[player_idx] += 1
        if action.upper() == Direction.SOUTH:
            self.counts[player_idx] -= 1

    def apply_actions(self):
        super(DummyInteractiveGame, self).apply_actions()
        self.counter += 1

    def get_state(self):
        state = super(DummyInteractiveGame, self).get_state()
        state["count"] = self.counter
        for i in range(self.num_players):
            state["player_{}_count".format(i)] = self.counts[i]
        return state


class OvercookedGame(Game):
    """
    Class for bridging the gap between Overcooked_Env and the Game interface

    Instance variable:
        - max_players (int): Maximum number of players that can be in the game at once
        - mdp (OvercookedGridworld): Controls the underlying Overcooked game logic
        - score (int): Current reward acheived by all players
        - max_time (int): Number of seconds the game should last
        - npc_policies (dict): Maps user_id to policy (Agent) for each AI player
        - npc_state_queues (dict): Mapping of NPC user_ids to LIFO queues for the policy to process
        - curr_tick (int): How many times the game server has called this instance's `tick` method
        - ticker_per_ai_action (int): How many frames should pass in between NPC policy forward passes.
            Note that this is a lower bound; if the policy is computationally expensive the actual frames
            per forward pass can be higher
        - action_to_overcooked_action (dict): Maps action names returned by client to action names used by OvercookedGridworld
            Note that this is an instance variable and not a static variable for efficiency reasons
        - human_players (set(str)): Collection of all player IDs that correspond to humans
        - npc_players (set(str)): Collection of all player IDs that correspond to AI
        - randomized (boolean): Whether the order of the layouts should be randomized

    Methods:
        - npc_policy_consumer: Background process that asynchronously computes NPC policy forward passes. One thread
            spawned for each NPC
        - _curr_game_over: Determines whether the game on the current mdp has ended
    """

    def __init__(
        self,
        layouts=["cramped_room"],
        mdp_params={},
        num_players=2,
        gameTime=30,
        playerZero="human",
        playerOne="human",
        showPotential=False,
        randomized=False,
        ticks_per_ai_action=1,
        **kwargs
    ):
        super(OvercookedGame, self).__init__(**kwargs)
        self.show_potential = showPotential
        self.mdp_params = mdp_params
        self.layouts = layouts
        self.max_players = int(num_players)
        self.mdp = None
        self.mp = None
        self.score = 0
        self.phi = 0
        self.max_time = min(int(gameTime), MAX_GAME_TIME)
        self.npc_policies = {}
        self.npc_state_queues = {}
        self.action_to_overcooked_action = {
            "STAY": Action.STAY,
            "UP": Direction.NORTH,
            "DOWN": Direction.SOUTH,
            "LEFT": Direction.WEST,
            "RIGHT": Direction.EAST,
            "SPACE": Action.INTERACT,
        }
        self.ticks_per_ai_action = ticks_per_ai_action
        self.curr_tick = 0
        self.human_players = set()
        self.npc_players = set()

        if randomized:
            random.shuffle(self.layouts)

        if playerZero != "human":
            player_zero_id = playerZero + "_0"
            self.add_player(player_zero_id, idx=0, buff_size=1, is_human=False)
            self.npc_policies[player_zero_id] = self.get_policy(
                playerZero, idx=0
            )
            self.npc_policies[player_zero_id].__init__(0)
            self.npc_state_queues[player_zero_id] = LifoQueue()

        if playerOne != "human":
            player_one_id = playerOne + "_1"
            self.add_player(player_one_id, idx=1, buff_size=1, is_human=False)
            self.npc_policies[player_one_id] = self.get_policy(
                playerOne, idx=1
            )
            self.npc_policies[player_one_id].__init__(1)
            self.npc_state_queues[player_one_id] = LifoQueue()
        # Always kill ray after loading agent, otherwise, ray will crash once process exits
        # Only kill ray after loading both agents to avoid having to restart ray during loading
        if ray.is_initialized():
            ray.shutdown()

        if kwargs["dataCollection"]:
            self.write_data = True
            self.write_config = kwargs["collection_config"]
        else:
            self.write_data = False

        self.trajectory = []

    def _curr_game_over(self):
        
        return time() - self.start_time >= self.max_time

    def needs_reset(self):
        return self._curr_game_over() and not self.is_finished()

    def add_player(self, player_id, idx=None, buff_size=-1, is_human=True):
        super(OvercookedGame, self).add_player(
            player_id, idx=idx, buff_size=buff_size
        )
        if is_human:
            self.human_players.add(player_id)
        else:
            self.npc_players.add(player_id)

    def remove_player(self, player_id):
        removed = super(OvercookedGame, self).remove_player(player_id)
        if removed:
            if player_id in self.human_players:
                self.human_players.remove(player_id)
            elif player_id in self.npc_players:
                self.npc_players.remove(player_id)
            else:
                raise ValueError("Inconsistent state")

    def npc_policy_consumer(self, policy_id):
        queue = self.npc_state_queues[policy_id]
        policy = self.npc_policies[policy_id]
        while self._is_active:
            state = queue.get()
            npc_action, _ = policy.action(state)
            super(OvercookedGame, self).enqueue_action(policy_id, npc_action)

    def is_full(self):
        return self.num_players >= self.max_players

    def is_finished(self):
        val = not self.layouts and self._curr_game_over()
        return val

    def is_empty(self):
        """
        Game is considered safe to scrap if there are no active players or if there are no humans (spectating or playing)
        """
        return (
            super(OvercookedGame, self).is_empty()
            or not self.spectators
            and not self.human_players
        )

    def is_ready(self):
        """
        Game is ready to be activated if there are a sufficient number of players and at least one human (spectator or player)
        """
        return super(OvercookedGame, self).is_ready() and not self.is_empty()

    def apply_action(self, player_id, action):
        pass

    def apply_actions(self):
        # Default joint action, as NPC policies and clients probably don't enqueue actions fast
        # enough to produce one at every tick
        joint_action = [Action.STAY] * len(self.players)

        # Synchronize individual player actions into a joint-action as required by overcooked logic
        for i in range(len(self.players)):
            # if this is a human, don't block and inject
            if self.players[i] in self.human_players:
                try:
                    # we don't block here in case humans want to Stay
                    joint_action[i] = self.pending_actions[i].get(block=False)
                except Empty:
                    pass
            else:
                # we block on agent actions to ensure that the agent gets to do one action per state
                joint_action[i] = self.pending_actions[i].get(block=True)

        # Apply overcooked game logic to get state transition
        prev_state = self.state
        self.state, info = self.mdp.get_state_transition(
            prev_state, joint_action
        )
        if self.show_potential:
            self.phi = self.mdp.potential_function(
                prev_state, self.mp, gamma=0.99
            )

        # Send next state to all background consumers if needed
        if self.curr_tick % self.ticks_per_ai_action == 0:
            for npc_id in self.npc_policies:
                self.npc_state_queues[npc_id].put(self.state, block=False)

        # Update score based on soup deliveries that might have occured
        curr_reward = sum(info["sparse_reward_by_agent"])
        self.score += curr_reward

        if (use200ScoreStopSoup and self.score >= 200):
            self.deactivate()


        transition = {
            "state": json.dumps(prev_state.to_dict()),
            "joint_action": json.dumps(joint_action),
            "reward": curr_reward,
            "time_left": max(self.max_time - (time() - self.start_time), 0),
            "score": self.score,
            "time_elapsed": time() - self.start_time,
            "cur_gameloop": self.curr_tick,
            "layout": json.dumps(self.mdp.terrain_mtx),
            "layout_name": self.curr_layout,
            "trial_id": str(self.start_time),
            "player_0_id": self.players[0],
            "player_1_id": self.players[1],
            "player_0_is_human": self.players[0] in self.human_players,
            "player_1_is_human": self.players[1] in self.human_players,
        }

        self.trajectory.append(transition)

        # Return about the current transition
        return prev_state, joint_action, info

    def enqueue_action(self, player_id, action):
        overcooked_action = self.action_to_overcooked_action[action]
        super(OvercookedGame, self).enqueue_action(
            player_id, overcooked_action
        )

    def reset(self):
        status = super(OvercookedGame, self).reset()
        if status == self.Status.RESET:
            # Hacky way of making sure game timer doesn't "start" until after reset timeout has passed
            self.start_time += self.reset_timeout / 1000

    def tick(self):
        self.curr_tick += 1
        return super(OvercookedGame, self).tick()

    def activate(self):
        super(OvercookedGame, self).activate()

        # Sanity check at start of each game
        if not self.npc_players.union(self.human_players) == set(self.players):
            raise ValueError("Inconsistent State")

        self.curr_layout = self.layouts.pop()
        self.mdp = OvercookedGridworld.from_layout_name(
            self.curr_layout, **self.mdp_params
        )
        global map_Name
        map_Name = self.curr_layout
        if self.show_potential:
            self.mp = MotionPlanner.from_pickle_or_compute(
                self.mdp, counter_goals=NO_COUNTERS_PARAMS
            )
        self.state = self.mdp.get_standard_start_state()
        if self.show_potential:
            self.phi = self.mdp.potential_function(
                self.state, self.mp, gamma=0.99
            )
        self.start_time = time()
        self.curr_tick = 0
        self.score = 0
        self.threads = []
        for npc_policy in self.npc_policies:
            self.npc_policies[npc_policy].reset()
            self.npc_state_queues[npc_policy].put(self.state)
            t = Thread(target=self.npc_policy_consumer, args=(npc_policy,))
            self.threads.append(t)
            t.start()

    def deactivate(self):
        super(OvercookedGame, self).deactivate()
        # Ensure the background consumers do not hang

        for npc_policy in self.npc_policies:
            self.npc_state_queues[npc_policy].put(self.state)

        # Wait for all background threads to exit
        for t in self.threads:
            t.join()

        # Clear all action queues
        self.clear_pending_actions()
        print("GAME OVER. Duration in ticks: " + str(self.curr_tick))

    def get_state(self):
        state_dict = {}
        state_dict["potential"] = self.phi if self.show_potential else None
        state_dict["state"] = self.state.to_dict()
        state_dict["score"] = self.score
        state_dict["time_left"] = max(
            self.max_time - (time() - self.start_time), 0
        )
        return state_dict

    def to_json(self):
        obj_dict = {}
        obj_dict["terrain"] = self.mdp.terrain_mtx if self._is_active else None
        obj_dict["state"] = self.get_state() if self._is_active else None
        return obj_dict

    def get_policy(self, npc_id, idx=0):
        #print("npc_id is " + str(npc_id))
        if npc_id.lower().startswith("rllib"):
            try:
                # Loading rllib agents requires additional helpers
                #print("Agent dir is" + str(AGENT_DIR))
                fpath = os.path.join(AGENT_DIR, npc_id, "agent")
                fix_bc_path(fpath)
                #print(fpath)
                agent = load_agent(fpath, agent_index=idx)
                return agent
            except Exception as e:
                raise IOError(
                    "Error loading Rllib Agent\n{}".format(e.__repr__())
                )
        else:
            try:
                fpath = os.path.join(AGENT_DIR, npc_id, "agent.pickle")
                #print(fpath)
                with open(fpath, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                raise IOError("Error loading agent\n{}".format(e.__repr__()))

    def get_data(self):
        """
        Returns and then clears the accumulated trajectory
        """
        data = {
            "uid": str(time()),
            "trajectory": self.trajectory,
        }
        self.trajectory = []
        # if we want to store the data and there is data to store
        if self.write_data and len(data["trajectory"]) > 0:
            configs = self.write_config
            # create necessary dirs
            data_path = create_dirs(configs, self.curr_layout)
            # the 3-layer-directory structure should be able to uniquely define any experiment
            with open(os.path.join(data_path, "result.pkl"), "wb") as f:
                pickle.dump(data, f)
        return data


class OvercookedTutorial(OvercookedGame):

    """
    Wrapper on OvercookedGame that includes additional data for tutorial mechanics, most notably the introduction of tutorial "phases"

    Instance Variables:
        - curr_phase (int): Indicates what tutorial phase we are currently on
        - phase_two_score (float): The exact sparse reward the user must obtain to advance past phase 2
    """

    def __init__(
        self,
        layouts=["tutorial_0"],
        mdp_params={},
        playerZero="human",
        playerOne="AI",
        phaseTwoScore=15,
        **kwargs
    ):
        super(OvercookedTutorial, self).__init__(
            layouts=layouts,
            mdp_params=mdp_params,
            playerZero=playerZero,
            playerOne=playerOne,
            showPotential=False,
            **kwargs
        )
        self.phase_two_score = phaseTwoScore
        self.phase_two_finished = False
        self.max_time = 0
        self.max_players = 2
        self.ticks_per_ai_action = 1
        self.curr_phase = 0
        # we don't collect tutorial data
        self.write_data = False

    @property
    def reset_timeout(self):
        return 1

    def needs_reset(self):
        if self.curr_phase == 0:
            return self.score > 0
        elif self.curr_phase == 1:
            return self.score > 0
        elif self.curr_phase == 2:
            return self.phase_two_finished
        return False

    def is_finished(self):
        return not self.layouts and self.score >= float("inf")

    def reset(self):
        super(OvercookedTutorial, self).reset()
        self.curr_phase += 1

    def get_policy(self, *args, **kwargs):
        return TutorialAI()

    def apply_actions(self):
        """
        Apply regular MDP logic with retroactive score adjustment tutorial purposes
        """
        _, _, info = super(OvercookedTutorial, self).apply_actions()

        human_reward, ai_reward = info["sparse_reward_by_agent"]

        # We only want to keep track of the human's score in the tutorial
        self.score -= ai_reward

        # Phase two requires a specific reward to complete
        if self.curr_phase == 2:
            self.score = 0
            if human_reward == self.phase_two_score:
                self.phase_two_finished = True


class DummyOvercookedGame(OvercookedGame):
    """
    Class that hardcodes the AI to be random. Used for debugging
    """

    def __init__(self, layouts=["cramped_room"], **kwargs):
        super(DummyOvercookedGame, self).__init__(layouts, **kwargs)

    def get_policy(self, *args, **kwargs):
        return DummyAI()


class DummyAI:
    """
    Randomly samples actions. Used for debugging
    """

    def __init__(self, id):
        self.id = id
        #print("Initialization")


    dirs =         [[0,1],[0,-1],[1,0],[-1,0]]
    oppositeDirs = [[0,-1],[0,1],[-1,0],[1,0]]

    # This needs to be changed if not the last version is used
    def GetRemainingTimeOfObject(self, object):
        cookingTime = object.cook_time
        if cookingTime == None:
            cookingTime = object.recipe.time

        if(object.cooking_tick==-1): # The soup has not started cooking yet
            return 100000

        return cookingTime - object.cooking_tick

    #End


    def getDirectionTo(self, x,y):
        
        if(self.distances[y][x] == 0):
            return None
        
        if(self.distances[y][x] > 5000):
            return None
        
        
        while self.distances[y][x] != 1:
            for i in range(0, len(self.dirs)):
                xNew = x+self.dirs[i][0]
                yNew = y+self.dirs[i][1]
                xNew2 = x+self.dirs[i][0]+self.dirs[i][0]
                yNew2 = y+self.dirs[i][1]+self.dirs[i][1]
                
                
                if self.insideBounds(xNew, yNew) and self.distances[yNew][xNew] == self.distances[y][x]-1 and self.isWalkable(xNew, yNew, False):
                    if self.distances[yNew2][xNew2] != 0:
                        if self.insideBounds(xNew2, yNew2) and self.distances[yNew2][xNew2] == self.distances[y][x]-2 and self.isWalkable(xNew2, yNew2, False):
                            x = xNew2
                            y = yNew2
                            break

                if self.insideBounds(xNew, yNew) and self.distances[yNew][xNew] == self.distances[y][x]-1 and self.isWalkable(xNew, yNew, False):
                    x = xNew
                    y = yNew
                    break
        #print([x,y])                
        for i in range(0, len(self.dirs)):
                xNew = x+self.dirs[i][0]
                yNew = y+self.dirs[i][1]
                if self.insideBounds(xNew, yNew) and self.distances[yNew][xNew] == self.distances[y][x]-1 and self.isWalkable(xNew, yNew, False):
                    #print(i)
                    return i

        #selectedDir = -1
        

        #if selectedDir==-1: # in case of no path
        #    return 1000000

        #x = x + self.dirs[selectedDir][0]
        #y = y + self.dirs[selectedDir][1]
        #if(self.distances[y][x] == 0):
        #    return selectedDir
        #selectedSecondDir = None
        #selectedPossibleDir = None

        #for i in range(0, len(self.dirs)):
        #    xNew = x+self.dirs[i][0]
        #    yNew = y+self.dirs[i][1]
        #    if self.insideBounds(xNew, yNew) and self.distances[yNew][xNew] == self.distances[y][x]-1 and self.isWalkable(xNew, yNew, False):
        #        if i == selectedDir:
        #            selectedSecondDir = i
        #        selectedPossibleDir = i
        #if selectedSecondDir == None:
        #    selectedSecondDir = selectedPossibleDir
        #return selectedSecondDir

    def distanceTo(self, x,y, ignoreAlly = False):
        if(self.distances[y][x] == 0):
            return 0
        if(self.distances[y][x] == 10000):
            return 10000


        selectedDir = -1
        for i in range(0, len(self.dirs)):
            xNew = x+self.dirs[i][0]
            yNew = y+self.dirs[i][1]
            if self.insideBounds(xNew, yNew) and self.distances[yNew][xNew] == self.distances[y][x]-1 and self.isWalkable(xNew, yNew, False):
                selectedDir = i

        if selectedDir==-1: # in case of no path
            return 10000

        x = x + self.dirs[selectedDir][0]
        y = y + self.dirs[selectedDir][1]
        if(self.distances[y][x] == 0):
            return 1
        if(self.distances[y][x] == 10000):
            return 10000

        selectedSecondDir = None
        selectedPossibleDir = -1
        for i in range(0, len(self.dirs)):
            xNew = x+self.dirs[i][0]
            yNew = y+self.dirs[i][1]
            if self.distances[yNew][xNew] == self.distances[y][x]-1 and self.isWalkable(xNew, yNew, False):
                if i == selectedDir:
                    selectedSecondDir = i
                selectedPossibleDir = i
        if selectedPossibleDir == -1:
            return 10000
        if selectedSecondDir == None:
            selectedSecondDir = selectedPossibleDir
        
        x = x + self.dirs[selectedSecondDir][0]
        y = y + self.dirs[selectedSecondDir][1]

        if selectedSecondDir == selectedDir:
            return 2 + self.distanceToHelper(x,y, ignoreAlly)
        else:
            return 3 + self.distanceToHelper(x,y, ignoreAlly)
        
    def distanceToHelper(self, x, y, ignoreAlly):
        if(self.distances[y][x] == 0):
            return 0
        selectedDir = -1
        for i in range(0, len(self.dirs)):
            xNew = x+self.dirs[i][0]
            yNew = y+self.dirs[i][1]
            if self.distances[yNew][xNew] == self.distances[y][x]-1 and self.isWalkable(xNew, yNew, ignoreAlly):
                selectedDir = i
        if(selectedDir == -1):
            return 10000
        x = x + self.dirs[selectedDir][0]
        y = y + self.dirs[selectedDir][1]
        return 1 + self.distanceToHelper(x,y,ignoreAlly)

    def managePositon(self, x, y, ignoreAlly, firstTime = False):
        

        for i in range(0, len(self.dirs)):
            posToConsider = [x+self.dirs[i][0], y + self.dirs[i][1]]
            if not self.insideBounds(posToConsider[0], posToConsider[1]):
                continue
            if not self.isWalkable(posToConsider[0], posToConsider[1], ignoreAlly):
                if not firstTime:
                    self.distances[posToConsider[1]][posToConsider[0]] = min(self.distances[y][x] + 1, self.distances[posToConsider[1]][posToConsider[0]])
                continue
            if not self.checkedPositions[posToConsider[1]][posToConsider[0]] == 0:
                continue

            self.positions.append(posToConsider)
            self.checkedPositions[posToConsider[1]][posToConsider[0]] = 1
            self.distances[posToConsider[1]][posToConsider[0]] = self.distances[y][x] + 1
                


    def printDistances(self):
        for x in range (0, len(self.distances)):
            print (self.distances[x])

    def computePathingFrom (self,x,y, ignoreAlly = False, useFirstTime = True):
        self.distances = [  [10000]*self.mapWidth for i in range(0, self.mapHeight)  ] 
        self.checkedPositions = [  [0]*self.mapWidth for i in range(0, self.mapHeight)  ] 

        self.distances[y][x] = 0
        self.checkedPositions[y][x] = 1

        index = 0
        self.positions = []
        self.positions.append([x,y])
        firstTime = useFirstTime
        while index < len(self.positions): 
            self.managePositon(self.positions[index][0], self.positions[index][1], ignoreAlly, firstTime)
            firstTime = False
            index = index + 1

            



    def isPlaceabale(self, x ,y):
        return self.map[y][x] == 'X'

    def isWalkable(self, x, y, ignoreAlly):
        if(ignoreAlly):
            return self.map[y][x] == ' ' or self.map[y][x] == 'x' or self.map[y][x] == chr(self.allyId+48) or self.map[y][x] == chr(self.id+48)
        else:
            return self.map[y][x] == ' ' or self.map[y][x] == chr(self.id+48)



    def initialization(self):
        try:
            self.init
        except AttributeError:

            ######## CUSTOMIZABLE PARAMETERS
            self.rangeToConsiderForAllyMovement = 1 # NOT USED ANYWHERE
            self.maximumWaitTimeForSoup = 3 # Might arrive to pick up sooner with 3 timesteps
            self.putSoupOnCounterBias = 1
            self.roomName = map_Name #The room to be loaded
            print("map name is " + str(self.roomName))
            self.biasInPuttingOnionsWhereThereAreMultipleOnions = 7
            #######  END OF CUSTOMIZABLE PARAMETERS

            self.init = None
            self.mdp = OvercookedGridworld.from_layout_name(
                self.roomName
            )
            self.allyId = 1 - self.id
            self.map = self.mdp.terrain_mtx

            #This can be used if an older version of overcooked is used,but this has changed
            #self.cookingTime = self.mdp.soup_cooking_time

            self.mapWidth = len(self.map[0])
            self.mapHeight = len (self.map)
            self.stoves = []
            self.onionDispensers = []
            self.delivery = []
            self.dishDispensers = []
            self.counters = []
            for y in range (0, len(self.map)):
                for x in range (0, len(self.map[y])):
                    if self.map[y][x] == 'P':
                        self.stoves.append([x,y])
                    if self.map[y][x] == 'S':
                        self.delivery.append([x,y])
                    if self.map[y][x] == 'O':
                        self.onionDispensers.append([x,y])
                    if self.map[y][x] == 'D':
                        self.dishDispensers.append([x,y])
                    if self.map[y][x] == 'X':
                        self.counters.append([x,y,False, 100000,100000]) # x,y, isSomethingOnIt, distance to closest delivery, distance to closest stove



    
    def insideBounds (self, x, y):
        if (x < 0 or y < 0):
            return False
        if (x >= self.mapWidth or y >= self.mapHeight):
            return False
        return True

    def takeVariablesFromState(self, state):
        self.x = state.players[self.id].position[0]
        self.y = state.players[self.id].position[1]
        self.orientation = state.players[self.id].orientation
        self.heldObject = state.players[self.id].held_object

        self.allyX = state.players[self.allyId].position[0]
        self.allyY = state.players[self.allyId].position[1]
        self.allyOrientation = state.players[self.allyId].orientation
        self.allyHeldObject= state.players[self.allyId].held_object

        #This exception happens only in the first go through this function
        try:
            self.map[self.allyLastY][self.allyLastX] = ' '
            self.map[self.lastY][self.lastX] = ' '
            self.map[self.lastXy][self.lastXx] = ' ' 
        except AttributeError:
            self.map

        self.lastY = self.y
        self.lastX = self.x
        self.allyLastY = self.allyY
        self.allyLastX = self.allyX
        
        self.map[self.allyY][self.allyX] = chr(self.allyId+48)

        orientationY = self.allyY+self.allyOrientation[1]
        orientationX = self.allyX+self.allyOrientation[0]
        if(self.map[orientationY][orientationX] == ' '):
            self.lastXx = orientationX
            self.lastXy = orientationY
            if (random.randint(0,100)<50):
                self.map[orientationY][orientationX] = 'x'

        self.map[self.y][self.x] = chr(self.id+48)




    def front(self):
        return [self.x +self.orientation[0], self.y + self.orientation[1]]

    def TakeOnionScore(self, x, y, state):
        if (self.distances[y][x]>5000):
            return 0

        if self.map[y][x] == 'O':
            return max(50 - self.distances[y][x], 0)
        if self.IsSoupReadyToStart(state, x, y):
            return 47
        if((x,y) in state.objects and state.objects[(x,y)].name == "onion"):
            return 50-self.distanceTo(x,y)

        # for i in range (0, len(self.onions)):
        #     if (self.onions[i].position[0] == x and self.onions[i].position[1] == y):
        #         myDist = 0
        #         self.computePathingFrom(x, y)
        #         myDist = myDist + self.distances[self.y][self.x]

        #         if self.getDirectionTo(self.x, self.y) == None:
        #            self.computePathingFrom(self.x, self.y, False, False)
        #            return 0


        #         expectedX = x + self.getDirectionTo(self.x, self.y)[0]
        #         expectedY = y + self.getDirectionTo(self.x, self.y)[1]



        #         closestDistanceToStove = 10000
        #         counterDist = 10000
        #         for j in range(0, len(self.stoves)):
        #             self.computePathingFrom(self.stoves[j][0], self.stoves[j][1])
        #             newDist = self.distances[expectedY][expectedX]
        #             newCounterDist = self.distances[y][x]
        #             if newDist < closestDistanceToStove:
        #                 closestDistanceToStove = newDist
        #             if newCounterDist < counterDist:
        #                 counterDist = newCounterDist
                
        #         myDist = myDist + closestDistanceToStove

        #         self.computePathingFrom(self.x, self.y, False, False)
            
        #         if(myDist > counterDist - self.putSoupOnCounterBias):
        #             return 0

        #         return max(48 - self.distances[y][x], 0)

        return 0

    def GoodSoupPlacementScore(self, x, y, state):
        #print("calculating score for " + str(x) + " " + str(y))
        self.computePathingFrom(self.x, self.y, False, False)
        for i in range(0, len(self.delivery)):
            if (self.delivery[i][0] == x and self.delivery[i][1] == y): # if it's a stove                                  
                    return max(50 - self.distances[y][x], 0)
                
        oneOtherPossibleLocation = False
        for i in range (0, len(self.counters)):
                xc = self.counters[i][0]
                yc = self.counters[i][1] 
                if (self.counters[i][3] < 5000 and (xc,yc) in state.objects and state.objects[(xc,yc)].name =="soup"):
                    oneOtherPossibleLocation = True

        if oneOtherPossibleLocation == True:
            return 0
        
        if(self.map[y][x]=='X' and not (x,y) in state.objects):
            for i in range (0, len(self.counters)):
                if (self.counters[i][0]==x and self.counters[i][1]==y):
                    return max(28 - self.counters[i][3], 0)

            return max(28 - self.distances[y][x], 0)



        return 0

    def GoodOnionPlacementScore(self, x, y, state):
        #print("calculating score for " + str(x) + " " + str(y))
        self.computePathingFrom(self.x, self.y, False, False)
        for i in range(0, len(self.stoves)):
            if (self.stoves[i][0] == x and self.stoves[i][1] == y): # if it's a stove
                if (not (x,y) in state.objects) or len(state.objects[(x,y)].ingredients) < 3:
                    
                    bias = 0
                    if (x,y) in state.objects:
                        bias = bias + len(state.objects[(x,y)].ingredients) * self.biasInPuttingOnionsWhereThereAreMultipleOnions
                            
                    return max(50 - self.distances[y][x], 0) + bias
                
        oneOtherPossibleLocation = False
        for i in range (0, len(self.counters)):
                xc = self.counters[i][0]
                yc = self.counters[i][1] 
                if (self.counters[i][4] < 5000 and (xc,yc) in state.objects and state.objects[(xc,yc)].name =="onion"):
                    oneOtherPossibleLocation = True

        if oneOtherPossibleLocation == True:
            return 0
        
        if(self.map[y][x]=='X' and not (x,y) in state.objects):
            for i in range (0, len(self.counters)):
                if (self.counters[i][0]==x and self.counters[i][1]==y):
                    return max(28 - self.counters[i][4], 0)
            # myDist = 0
            # self.computePathingFrom(x, y)
            # myDist = myDist + self.distances[self.y][self.x]

            # try:

            #     if (self.getDirectionTo(self.x, self.y) == None): # if there is no path
            #         self.computePathingFrom(self.x, self.y,False, False)
            #         return 0

            #     expectedX = x + self.getDirectionTo(self.x, self.y)[0]
            #     expectedY = y + self.getDirectionTo(self.x, self.y)[1]
            # except TypeError:
            #     self.computePathingFrom(self.x, self.y,False, False)
            #     return 0

            # closestDistanceToStove = 10000
            # counterDist = 10000
            # for j in range(0, len(self.stoves)):
            #     self.computePathingFrom(self.stoves[j][0], self.stoves[j][1])
            #     newDist = self.distances[expectedY][expectedX]
            #     newCounterDist = self.distances[y][x]
            #     if newDist < closestDistanceToStove:
            #         closestDistanceToStove = newDist
            #     if newCounterDist < counterDist:
            #         counterDist = newCounterDist
            
            # myDist = myDist + closestDistanceToStove

            # self.computePathingFrom(self.x, self.y, False, False)
        
            # if(myDist < counterDist + self.putSoupOnCounterBias):
            #     return 0

            return max(28 - self.distances[y][x], 0)



        return 0

    def actionFromDir(self, dir):
        if dir == 0:
            return Direction.SOUTH
        if dir == 1:
            return Direction.NORTH
        if dir == 2:
            return Direction.EAST
        if dir == 3:
            return Direction.WEST
        return Action.STAY
    
    def oppositeActionFromDir(self, dir):
        if dir == 0:
            return Direction.NORTH
        if dir == 1:
            return Direction.SOUTH
        if dir == 2:
            return Direction.WEST
        if dir == 3:
            return Direction.EAST
        return Action.STAY

    def TimeToPickUpAndDeliverSoup(self, state, x, y, heldObject):
        if (heldObject!= None and heldObject.name == "soup"):
            return [10000,1,1]
        
        time = 0

        self.computePathingFrom(x,y,False, False)

        closestCounter = 10000
        bestX = -1
        bestY = -1
        hasOnion = False
        hasNothing = False

        timeTilCounter = 0
        timeTilDish = 0
        TimeTilSoup = 0

        counterx = -1
        countery = -1
        dishx = -1
        dishy = -1
        soupx = -1
        soupy = -1


        #print ("x initial: " + str(x) + " y initial " + str(y))
        if (heldObject != None and heldObject.name == "onion"):
            time = time + 1 # adding the time to perform the action of dropping the onion
            hasOnion = True
            for i in range(0, len(self.stoves)):
                if(self.IsSoupMissingOnions(state, i)):
                    stoveX = self.stoves[i][0]
                    stoveY = self.stoves[i][1]
                    if self.distances[stoveY][stoveX] < closestCounter:
                        closestCounter = self.distances[stoveY][stoveX]
                        bestX = stoveX
                        bestY = stoveY


            #self.printDistances()
            for i in range(0, len(self.counters)):
                counterX = self.counters[i][0]
                counterY = self.counters[i][1]
                if((counterX, counterY ) not in state.objects ): # if the counter is empty
                    if self.distances[counterY][counterX] < closestCounter:
                        closestCounter = self.distances[counterY][counterX]
                        bestX = counterX
                        bestY = counterY

            timeTilCounter = closestCounter
            counterx = bestX
            countery = bestY
            if bestX == -1:
                #print ("crying, there is no place to put the onion")
                return [10000,1,1] # cry, there is no place to put the onion, and we cry because we hold the onion and there is nothing we can do
            time = time + closestCounter
            
            bestDist = 5000

            for i in range (0, len(self.dirs)):
                newX = bestX + self.dirs[i][0]
                newY = bestY + self.dirs[i][1]
                if not self.insideBounds(newX, newY) or not self.isWalkable(newX,newY, True):
                    continue
                if(self.distances[newY][newX] < bestDist):
                    bestDist = self.distances[newY][newX]
                    x = newX
                    y = newY

        closestDish = 10000
        bestX = -1
        bestY = -1

        self.computePathingFrom(x,y,False, False) # x and y might have new values
        #self.printDistances()
        if(heldObject == None or hasOnion):
            time = time + 1 # adding the time to perform the action of picking the dish
            hasNothing = True
            for i in range (0, len(self.dishDispensers)):
                dishX = self.dishDispensers[i][0]
                dishY = self.dishDispensers[i][1]
                if(self.distances[dishY][dishX] < closestDish):
                    closestDish = self.distances[dishY][dishX]
                    bestX = dishX
                    bestY = dishY

            
            for i in range(0, len(self.counters)):
                dishX = self.counters[i][0]
                dishY = self.counters[i][1]
                if((dishX, dishY ) in state.objects and state.objects[(dishX, dishY )].name == "dish"): # if there is a dish on a counter
                    if self.distances[dishY][dishX] < closestDish:
                        closestDish = self.distances[dishY][dishX]
                        bestX = dishX
                        bestY = dishY

            timeTilDish = closestDish
            dishx = bestX
            dishy = bestY

            if bestX == -1:
                #print ("crying, there is no place to take dish from")
                return [10000,1,1] # cry
            time = time + closestDish
            
            bestDist = 10000
            for i in range (0, len(self.dirs)):
                newX = bestX + self.dirs[i][0]
                newY = bestY + self.dirs[i][1]
                if not self.insideBounds(newX, newY) or not self.isWalkable(newX,newY, True):
                    continue
                if(self.distances[newY][newX] < bestDist):
                    bestDist = self.distances[newY][newX]
                    x = newX
                    y = newY

        self.computePathingFrom(x,y,False, False) # x and y might have different values

        if (len(self.startedSoups) == 0): # there is no soup to be picked up
            return [10000,1,1]

        bestDist = 10000
        bestX = -1
        bestY = -1
        #self.printDistances()
        for i in range(0, len(self.startedSoups)):
            newX = self.startedSoups[i].position[0]
            newY = self.startedSoups[i].position[1]
            
            if(state.objects[(newX, newY)].cook_time_remaining < bestDist):
                bestDist = state.objects[(newX, newY)].cook_time_remaining
                bestX = newX
                bestY = newY

        if bestX == -1:
            #print ("crying, there is no soup to take")
            return [10000,1,1] # cry

        bestDist = 10000
        for i in range (0, len(self.dirs)):
            newX = bestX + self.dirs[i][0]
            newY = bestY + self.dirs[i][1]
            if not self.insideBounds(newX, newY) or not self.isWalkable(newX,newY, True):
                    continue
            if(self.distances[newY][newX] < bestDist):
                bestDist = self.distances[newY][newX]
                x = newX
                y = newY


        #Time until taking the soup, soup location
        return [time + bestDist, [bestX, bestY], [ [timeTilCounter,(counterx, countery)], [timeTilDish, (dishx, dishy)] ] ]

    def FindTargetForDelivery(self, state, x, y):
        self.computePathingFrom(x,y, True, False)
        canDeliverToDelivery = False

        bestDelivery = 10000

        bestX = -1
        bestY = -1
        for i in range(0, len(self.delivery)):
            dist = self.distances[self.delivery[i][1]][self.delivery[i][0]]
            if dist<bestDelivery:
                bestX = self.delivery[i][0]
                bestY = self.delivery[i][1]
                canDeliverToDelivery = True

        if canDeliverToDelivery:
            return [bestX, bestY]
        
        bestCounter = 10000

        for i in range (0, len(self.counters)):
            if (self.distances[self.counters[i][1]][self.counters[i][0]]) > 4000:
                continue
            if (self.counters[i][3] < bestCounter):
                bestX = self.counters[i][0]
                bestY = self.counters[i][1]

        return [bestX, bestY]


    def TimeToPickUpSoup(self, state, x, y):
        print ("Time to pick up")

    def canDeliverSoup(self, state):

        self.computePathingFrom(self.x, self.y, True, False)

        for j in range (0, len(self.delivery)):
            dist = self.distances[self.delivery[j][1]][self.delivery[j][0]]
            if dist<5000:
                return True


    def decideMyBehaviour(self, state):
        self.decidedAction = Action.STAY
        
        self.behaviour = "BringOnion"


        [timeUntilICanDeliverSoup, mySoupLocation, _]  = self.TimeToPickUpAndDeliverSoup(state,self.x,self.y, self.heldObject)
        [timeUntilAllyCanDeliverSoup, allySoupLocation, _]  = self.TimeToPickUpAndDeliverSoup(state,self.allyLastX, self.allyY, self.allyHeldObject)

        

        if timeUntilICanDeliverSoup < 5000:
            soupAtPos = state.objects[(mySoupLocation[0], mySoupLocation[1])]
            timeUntilSoupIsDone = soupAtPos.cook_time_remaining
            #print("TIME UNTIL SOUP IS DONE IS " + str(timeUntilSoupIsDone))
            if (mySoupLocation == allySoupLocation):
                if (timeUntilICanDeliverSoup < timeUntilAllyCanDeliverSoup):
                    if (timeUntilICanDeliverSoup >= timeUntilSoupIsDone):
                        self.behaviour = "PickUpSoup"
                elif (timeUntilAllyCanDeliverSoup == timeUntilICanDeliverSoup):
                    if(self.id == 0): # in case of a tie, the lower id will try to bring the soup, this is only for when playing with another ai
                        if (timeUntilICanDeliverSoup >= timeUntilSoupIsDone):
                            self.behaviour = "PickUpSoup"

            else:
                if (timeUntilICanDeliverSoup < 4000):
                    if (timeUntilICanDeliverSoup >= timeUntilSoupIsDone):
                        self.behaviour = "PickUpSoup"


        #print("My time to pick and deliver soup: " + str(timeUntilICanDeliverSoup))
        #print("Ally time to pick and deliver soup: " + str(timeUntilAllyCanDeliverSoup))

        

        for i in range (0, len (self.startedSoups)):
            if self.startedSoups[i].cook_time_remaining  > 10:
                continue
            newX = self.startedSoups[i].position[0]
            newY = self.startedSoups[i].position[1]
            self.computePathingFrom(newX, newY, True, True)
            isThereDishForSoup = False
            for j in range(0, len(self.dishDispensers)):
                x = self.dishDispensers[j][0]
                y = self.dishDispensers[j][1]
                if( self.distances[y][x] < 4000):
                    isThereDishForSoup = True
                    break
            for j in range (0, len(self.dishes)):
                x = self.dishes[j].position[0]
                y = self.dishes[j].position[1]
                if( self.distances[y][x] < 4000):
                    isThereDishForSoup = True
                    break
            if isThereDishForSoup == False:
                self.computePathingFrom(self.x, self.y, True, False)
                isThereDishNearMe = False
                for j in range(0, len(self.dishDispensers)):
                    x = self.dishDispensers[j][0]
                    y = self.dishDispensers[j][1]
                    if( self.distances[y][x] < 4000):
                        isThereDishNearMe = True
                        break
                for j in range (0, len(self.dishes)):
                    x = self.dishes[j].position[0]
                    y = self.dishes[j].position[1]
                    if( self.distances[y][x] < 4000):
                        isThereDishNearMe = True
                        break
                if isThereDishNearMe and( self.heldObject == None or self.heldObject.name == "dish"):
                    self.behaviour = "BringPlate"

            
        if self.heldObject != None:
            if(self.heldObject.name == "soup"):
                self.behaviour = "DeliverSoup"


    def IsSoupMissingOnions(self, state, i):
        x=self.stoves[i][0]
        y=self.stoves[i][1]
        if (x,y) in state.objects and len(state.objects[(x,y)].ingredients) == 3:
            return False
                
        return True

    def IsSoupReadyToStart(self, state, x, y):
        for i in range(0, len(self.stoves)):
            if (self.stoves[i][0] == x and self.stoves[i][1] == y): # if we found the correct stove
                if (x,y) in state.objects and len(state.objects[(x,y)].ingredients) == 3 and state.objects[(x,y)].is_cooking == False : # here we can check if it's the correct order
                    if state.objects[(x,y)].is_ready:
                        return False
                    return True
                
        return False
        

    def takeAction(self, state):

        if random.randint(1, 10) == 1:
            return Action.STAY

        self.computePathingFrom(self.x, self.y, False, False)

        if  self.heldObject == None and self.IsSoupReadyToStart(state, self.front()[0], self.front()[1]):
            #print("doing here1")
            return Action.INTERACT

        #print("Behavior is: " + self.behaviour)
        
        if self.behaviour == "BringOnion":

            if(self.heldObject == None):
                if  self.IsSoupReadyToStart(state, self.front()[0], self.front()[1]):
                    #print("doing here2")
                    return Action.INTERACT
                
                bestScore = 0
                targetX = None
                targetY = None
                for y in range (0, len(self.map)):
                    for x in range (0, len(self.map[0])):
                        newScore = self.TakeOnionScore(x,y, state) 
                        #print(str(x) + " " + str(y) + " Obtained " + str(newScore))
                        if  newScore > bestScore:
                            bestScore = newScore
                            targetX = x
                            targetY = y
                if self.front()[0]== targetX and self.front()[1]==targetY:
                    #print("doing here3")
                    return Action.INTERACT
                
                for i in range (0, len(self.dirs)):
                    if self.x + self.dirs[i][0] == targetX and self.y+self.dirs[i][1] == targetY:
                        return self.actionFromDir(i)
                    



                if(targetX != None ):
                    return self.oppositeActionFromDir(self.getDirectionTo(targetX,targetY))


            if (self.heldObject != None):
                if(self.heldObject.name == "onion"):
                    bestScore = 0
                    targetX = None
                    targetY = None
                    for y in range (0, len(self.map)):
                        for x in range (0, len(self.map[0])):
                            newScore = self.GoodOnionPlacementScore(x,y, state) 
                            #print("Obtained " + str(newScore))
                            if  newScore > bestScore:
                                bestScore = newScore
                                targetX = x
                                targetY = y
                    if self.front()[0]== targetX and self.front()[1]==targetY:
                        #print("doing here4")
                        return Action.INTERACT
                    
                    for i in range (0, len(self.dirs)):
                        if self.x + self.dirs[i][0] == targetX and self.y+self.dirs[i][1] == targetY:
                            return self.actionFromDir(i)
                        



                    if(targetX != None ):
                        return self.oppositeActionFromDir(self.getDirectionTo(targetX,targetY))
        
        if self.behaviour == "BringPlate":
            targetX = -1
            targetY = -1
            if self.heldObject == None:
                
                bestDist = 10000
                self.computePathingFrom(self.x, self.y, False, False)
                for j in range(0, len(self.dishDispensers)):
                    x = self.dishDispensers[j][0]
                    y = self.dishDispensers[j][1]
                    if( self.distances[y][x] < bestDist):
                        targetX = x
                        targetY = y
                        bestDist = self.distances[y][x]
                        break
                for j in range (0, len(self.dishes)):
                    x = self.dishes[j].position[0]
                    y = self.dishes[j].position[1]
                    if( self.distances[y][x]-5 < bestDist):
                        targetX = x
                        targetY = y
                        bestDist = self.distances[y][x]-5
                        break

            if self.heldObject != None and self.heldObject.name == "dish":
                self.computePathingFrom(self.x, self.y, False, False)
                bestDist = 10000
                for i in range (0, len(self.counters)):
                    x = self.counters[i][0]
                    y = self.counters[i][1]
                    if (x,y) not in state.objects and self.distances[y][x] < bestDist:
                        bestDist = self.distances[y][x]
                        targetX = x
                        targetY = y


            if self.front()[0]== targetX and self.front()[1]==targetY:
                return Action.INTERACT
            
            for i in range (0, len(self.dirs)):
                if self.x + self.dirs[i][0] == targetX and self.y+self.dirs[i][1] == targetY:
                    return self.actionFromDir(i)
                
            return self.oppositeActionFromDir(self.getDirectionTo(targetX,targetY))
        


        if self.behaviour == "PickUpSoup":
            #self.printDistances()
            [timeTilSoup, soupLocation, params] =  self.TimeToPickUpAndDeliverSoup(state,self.x,self.y, self.heldObject)
            self.computePathingFrom(self.x, self.y, False, False)
            #print ([timeTilSoup, soupLocation, params])

            if self.heldObject != None and self.heldObject.name == "onion":
                targetX = params[0][1][0]
                targetY = params[0][1][1]

            if self.heldObject == None:
                targetX = params[1][1][0]
                targetY = params[1][1][1]

            if self.heldObject != None and self.heldObject.name == "dish":
                targetX = soupLocation[0]
                targetY = soupLocation[1]

            #print ("Target for delivery is " + str(targetX) + " " + str(targetY))

            if self.front()[0]== targetX and self.front()[1]==targetY:
                return Action.INTERACT
            
            for i in range (0, len(self.dirs)):
                if self.x + self.dirs[i][0] == targetX and self.y+self.dirs[i][1] == targetY:
                    return self.actionFromDir(i)
            #print("we got here...")
            return self.oppositeActionFromDir(self.getDirectionTo(targetX,targetY))
        
        if self.behaviour == "DeliverSoup":
            bestScore = 0
            targetX = None
            targetY = None
            for y in range (0, len(self.map)):
                for x in range (0, len(self.map[0])):
                    newScore = self.GoodSoupPlacementScore(x,y, state) 
                    #print("Obtained " + str(newScore))
                    if  newScore > bestScore:
                        bestScore = newScore
                        targetX = x
                        targetY = y
            if self.front()[0]== targetX and self.front()[1]==targetY:
                return Action.INTERACT
            
            for i in range (0, len(self.dirs)):
                if self.x + self.dirs[i][0] == targetX and self.y+self.dirs[i][1] == targetY:
                    return self.actionFromDir(i)
                



            if(targetX != None ):
                return self.oppositeActionFromDir(self.getDirectionTo(targetX,targetY))
        return Action.STAY
            #if(self.heldObject.name == "onion"):
                

    def checkObjects(self, state):

        self.soups = []
        self.startedSoups = []
        self.dishes = []
        self.onions = []
        for i in range (0, len(self.counters)):
            self.counters[i][2] = False
            self.counters[i][3] = 100000


        for i in range(0, len(state.objects.keys())):
            obj = state.objects[list(state.objects.keys())[i]]
            if(obj.name == "soup"):
               self.soups.append(obj)
               if(obj.is_cooking or obj.is_ready):
                   self.startedSoups.append(obj)
            if(obj.name == "dish"):
                self.dishes.append(obj)
            if(obj.name == "onion"):
                self.onions.append(obj)

            for i in range (0, len(self.counters)):
                if (self.counters[i][0] == obj.position[0] and self.counters[i][1] == obj.position[1] ):
                    self.counters[i][2] = True

        for i in range(0, len(self.counters)):
            if(self.counters[i][2] == True):
                continue
            self.computePathingFrom(self.counters[i][0], self.counters[i][1], True)


            bestDelivery = 100000
            for j in range (0, len(self.delivery)):
                dist = self.distances[self.delivery[j][1]][self.delivery[j][0]]
                if dist<bestDelivery:
                    bestDelivery = dist
            self.counters[i][3] = bestDelivery

            bestDelivery = 100000
            for j in range (0, len(self.stoves)):
                dist = self.distances[self.stoves[j][1]][self.stoves[j][0]]
                if dist<bestDelivery:
                    bestDelivery = dist
            self.counters[i][4] = bestDelivery

        #obj = state.objects["onion"]
        #print(obj)
        


    def action(self, state):
        self.initialization()
        self.takeVariablesFromState(state)

        #for x in range (0, len(self.map)):
            #print (self.map[x])
        
        #print("Tking action knowing " + str(state))


        self.computePathingFrom(self.x, self.y)
        self.checkObjects(state)
        
        #print("STARTED SOUPS:")
        #print(self.startedSoups)

        self.decideMyBehaviour(state)
        
        action = self.takeAction(state)
        #print("Doing " + str(action)) 

        

        return action, None


        [action] = random.sample(
            [
                Action.STAY,
                Direction.NORTH,
                Direction.SOUTH,
                Direction.WEST,
                Direction.EAST,
                Action.INTERACT,
            ],
            1,
        )
        return action, None

    def reset(self):
        pass


class DummyComputeAI(DummyAI):
    """
    Performs simulated compute before randomly sampling actions. Used for debugging
    """

    def __init__(self, compute_unit_iters=1e5):
        """
        compute_unit_iters (int): Number of for loop cycles in one "unit" of compute. Number of
                                    units performed each time is randomly sampled
        """
        super(DummyComputeAI, self).__init__()
        self.compute_unit_iters = int(compute_unit_iters)

    def action(self, state):
        # Randomly sample amount of time to busy wait
        iters = random.randint(1, 10) * self.compute_unit_iters

        # Actually compute something (can't sleep) to avoid scheduling optimizations
        val = 0
        for i in range(iters):
            # Avoid branch prediction optimizations
            if i % 2 == 0:
                val += 1
            else:
                val += 2

        # Return randomly sampled action
        return super(DummyComputeAI, self).action(state)


class StayAI:
    """
    Always returns "stay" action. Used for debugging
    """

    def action(self, state):
        return Action.STAY, None

    def reset(self):
        pass


class TutorialAI:
    COOK_SOUP_LOOP = [
        # Grab first onion
        Direction.WEST,
        Direction.WEST,
        Direction.WEST,
        Action.INTERACT,
        # Place onion in pot
        Direction.EAST,
        Direction.NORTH,
        Action.INTERACT,
        # Grab second onion
        Direction.WEST,
        Action.INTERACT,
        # Place onion in pot
        Direction.EAST,
        Direction.NORTH,
        Action.INTERACT,
        # Grab third onion
        Direction.WEST,
        Action.INTERACT,
        # Place onion in pot
        Direction.EAST,
        Direction.NORTH,
        Action.INTERACT,
        # Cook soup
        Action.INTERACT,
        # Grab plate
        Direction.EAST,
        Direction.SOUTH,
        Action.INTERACT,
        Direction.WEST,
        Direction.NORTH,
        # Deliver soup
        Action.INTERACT,
        Direction.EAST,
        Direction.EAST,
        Direction.EAST,
        Action.INTERACT,
        Direction.WEST,
    ]

    COOK_SOUP_COOP_LOOP = [
        # Grab first onion
        Direction.WEST,
        Direction.WEST,
        Direction.WEST,
        Action.INTERACT,
        # Place onion in pot
        Direction.EAST,
        Direction.SOUTH,
        Action.INTERACT,
        # Move to start so this loops
        Direction.EAST,
        Direction.EAST,
        # Pause to make cooperation more real time
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
    ]

    def __init__(self):
        self.curr_phase = -1
        self.curr_tick = -1

    def action(self, state):
        self.curr_tick += 1
        if self.curr_phase == 0:
            return (
                self.COOK_SOUP_LOOP[self.curr_tick % len(self.COOK_SOUP_LOOP)],
                None,
            )
        elif self.curr_phase == 2:
            return (
                self.COOK_SOUP_COOP_LOOP[
                    self.curr_tick % len(self.COOK_SOUP_COOP_LOOP)
                ],
                None,
            )
        return Action.STAY, None

    def reset(self):
        self.curr_tick = -1
        self.curr_phase += 1
