import pickle
import random
import json
from collections import deque
from os.path import isfile
import numpy as np
from keras.models import clone_model, load_model, Model
from keras.layers import Dense, Activation, Input

IN_GAME = False


class Basic():
    def __init__(self):
        self.observation = None
        self.value = None
        self.done = None
        self.a = None
        self.reset()

    def reset(self):
        self.value = random.randint(1, 10)
        v = self.value
        self.observation = [v, v, v]
        self.done = False
        print(self.observation)

    def getStateSize(self):
        return 3

    def getState(self):
        return self.observation

    def act(self, a):
        self.a = a

    def step(self):
        if self.a == 0:
            self.value = max(1, self.value - 1)
        if self.a == 1:
            self.value = min(10, self.value + 1)
        v = self.value
        minv = self.observation[0]
        maxv = self.observation[2]
        self.observation[0] = min(minv, v)
        self.observation[1] = v
        self.observation[2] = max(maxv, v)
        print(self.observation)
        if self.observation[0] == 1 and self.observation[2] == 10:
            self.done = True
        return self.done

    def getReward(self):
        if self.done:
            return 1.
        else:
            return 0.


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
            self.env = Basic()

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
                          'update_interval': 1000,
                          'end_e': 0.01,
                          'start_e': 0.2,
                          'step_end_e': 1000,
                          'exp_len': 1000000,
                          'team': team
                          }

            # connect to server
            self.env = Basic()

            # set input_dim
            self.state['input_dim'] = self.env.getStateSize()

            self.state['actions'] = [0, 1]
            self.state['output_dim'] = len(self.state['actions'])

            for i in range(self.state['output_dim']):
                self.state['net_files'].append('net' + str(i) + '.h5')
                self.state['tnet_files'].append('target_net' + str(i) + '.h5')

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

    def _train(self, exp_sample):
        states = []
        targets = []
        for (s, a, r, s1, done) in exp_sample:
            states.append(s)
            if not done:
                qs = [q.predict(np.array([s1])) for q in self.qs]
                maxa_idx = qs.index(max(qs))
                maxq = self.qs1[maxa_idx].predict(np.array([s1]))
                maxq = maxq[0][0]
                targets.append(r + self.state['g'] * maxq)
            else:
                targets.append(r)
        q = self.qs[self.state['actions'].index(a)]
        q.fit(np.array(states), targets, batch_size=self.state['batch_size'], epochs=self.state['epochs'],
              verbose=0)

    def run(self, episodes, test=False):
        eps = 0
        steps = 0
        rewards = 0.
        while True:
            steps += 1
            s = self.env.getState()
            if random.uniform(0, 1) <= self._e():
                a = random.choice(self.state['actions'])
            # policy action
            else:
                qs = [q.predict(np.array([s])) for q in self.qs]
                a = self.state['actions'][qs.index(max(qs))]
            self.env.act(a)
            done = self.env.step()
            done1 = done
            s1 = self.env.getState()
            r = self.env.getReward()
            rewards += r
            self.exp.append((s, a, r, s1, done))
            if self.exp:
                idx = np.random.choice(len(self.exp), min(self.state['batch_size'], len(self.exp)))
                exp_sample = [self.exp[i] for i in idx]
                self._train(exp_sample)
                self.state['step'] += 1
            if self.state['step'] % self.state['update_interval'] == 0 and not self.state['step'] == 0:
                self._clone()
                print('Step ' + str(self.state['step']) + ', network cloned.')
            if not done == done1:
                print('Really!?')
            if done:
                self.state['episode'] += 1
                eps += 1
                print('Steps =', steps, 'Episode', str(eps), 'done!')
                rewards = 0.
                steps = 0
                self.env.reset()

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
        if self.state['step'] >= self.state['step_end_e']:
            return self.state['end_e']
        else:
            return self.state['start_e'] - (self.state['start_e'] - self.state['end_e']) / self.state['step_end_e'] * \
                   self.state['step']


def main():
    file = 'state.json'
    if isfile(file):
        ag = Agent(file=file)
    else:
        ag = Agent('base_left')
    ag.run(2)
    ag.save()


if __name__ == "__main__":
    main()
