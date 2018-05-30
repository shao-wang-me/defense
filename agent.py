import pickle
import random
import json
from collections import deque
from os.path import isfile
from keras.models import clone_model, load_model, Model
from keras.layers import Dense, Activation, Input

from hfo import *


class Agent:
    def __init__(self, team=None, file=None):
        if file:  # load
            # load state
            with open(file) as f:
                self.state = json.load(f)
            # load networks
            self.qs = []
            self.qs1 = []
            for i in range(self.state['output_dim']):
                self.qs.append(load_model(self.state['net_file'][i]))
                self.qs1.append(load_model(self.state['tnet_files'][i]))
            self.qs[0].summary()
            # load experience
            with open(self.state['exp_file']) as f:
                self.exp = pickle.load(f)
            # connect to server
            self.env = HFOEnvironment()
            self.env.connectToServer(feature_set=HIGH_LEVEL_FEATURE_SET, team_name=self.state['team'])
        else:  # new
            # initialize state
            self.state = {'state_file': 'state.json',
                          'exp_file': 'experience',
                          'net_files': [],
                          'tnet_files': [],
                          'episode': 0,
                          'step': 0,
                          'g': 0.99,
                          'batch_size': 32,
                          'epochs': 1,
                          'update_interval': 10000,
                          'end_e': 0.01,
                          'start_e': 1.,
                          'step_end_e': 1000000,
                          'exp_len': 1000000,
                          'team': team
                          }
            if team == 'base_left':
                self.state['actions'] = [DRIBBLE, SHOOT]
            elif team == 'base_right':
                self.state['actions'] = [MOVE, DEFEND_GOAL, REDUCE_ANGLE_TO_GOAL, GO_TO_BALL, MARK_PLAYER]
            self.state['output_dim'] = len(self.state['actions'])

            for i in range(self.state['output_dim']):
                self.state['net_files'].append('net' + str(i) + '.h5')
                self.state['tnet_files'].append('target_net' + str(i) + '.h5')

            # connect to server
            self.env = HFOEnvironment()
            self.env.connectToServer(feature_set=HIGH_LEVEL_FEATURE_SET, team_name=team)

            # set input_dim
            self.state['input_dim'] = self.env.getStateSize()

            # initialize networks
            # input: state
            # output: Q value for each action
            inputs = Input(shape=(self.state['input_dim'],))
            x = Dense(units=20)(inputs)
            x = Dense(units=20)(x)
            self.qs = []
            for i in range(self.state['output_dim']):
                predictions = Dense(units=1)(x)
                self.qs.append(Model(inputs=inputs, outputs=predictions))
                self.qs[i].compile(optimizer='adam', loss='mean_squared_error')
            self._clone()
            self.qs[0].summary()

            # initialize experience
            self.exp = deque(maxlen=self.state['exp_len'])

    def run(self, episodes, test=False):
        for i in range(episodes):
            status = IN_GAME
            while status == IN_GAME:
                s = self.env.getState()
                # random action
                if random.uniform(0, 100) <= self._e():
                    a = random.choice(self.state['actions'])
                # policy action
                else:
                    qs = [q.predict(np.array([s])) for q in self.qs]
                    print(qs)
                    a = self.state['actions'][qs.index(max(qs))]
                self.env.act(a)
                status = self.env.step()
                s1 = self.env.getState()
                r = self._reward(status)
                self._push(self.exp, (s, a, r, s1, status))
                exp_sample = np.random.choice(self.exp, min(self.state['batch_size'], len(self.exp)))
                states = []
                targets = []
                for (s, a, r, s1, status) in exp_sample:
                    states.append(s)
                    if status == IN_GAME:
                        maxq = max([q.predict(np.array(s1)) for q in self.qs1])
                        targets.append(r + self.state['g'] * maxq)
                    else:
                        targets.append(r)
                q = self.qs[self.state['actions'].index(a)]
                q.fit(states, targets, batch_size=self.state['batch_size'], epochs=self.state['epochs'])
                self.state['step'] += 1
                if self.state['step'] % self.state['update_interval'] == 0:
                    self._clone()
        self.state['episode'] += 1

    def save(self):
        # save state
        with open(self.state['state_file']) as f:
            json.dump(self.state, f)
        # save networks
        for i in range(self.state['output_dim']):
            q = self.qs[i]
            q1 = self.qs1[i]
            q.save(self.state['net_file'][i])
            q1.save(self.state['tnet_files'][i])
        # save experience
        with open(self.state['exp_file']) as f:
            pickle.dump(self.exp, f)

    def _clone(self):
        self.qs1 = []
        for q in self.qs:
            q1 = clone_model(q)
            q1.set_weights(q.get_weights())
            self.qs1.append(q1)

    def _e(self):
        if self.state['episode'] >= self.state['step_end_e']:
            return self.state['end_e']
        else:
            return self.state['start_e'] - (self.state['start_e'] - self.state['end_e']) / self.state['step_end_e'] * \
                   self.state['episode']

    @staticmethod
    def _reward(status):
        return 1. if status == GOAL else 0

    @staticmethod
    def _push(xs, x):
        xs[1:] = xs[:-1]
        xs[0] = x


def main():
    file = 'state.json'
    if isfile(file):
        ag = Agent(file=file)
    else:
        ag = Agent('base_left')
    ag.run(100)
    ag.save()


if __name__ == "__main__":
    main()
