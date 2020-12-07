Deepmind의 DQN(Deep Q Network)을 이용하여 벽돌깨기 게임 학습시키기
=========
2160021 전영우, 22000035 권민혁
 
프로젝트를 하게 된 계기
-------
알파고를 만든 회사인 DeepMind의 초기 알고리즘인 DQN(Deep Q Network)은, 단 몇 시간 만에 현존하는 최고의 벽돌깨기 게이머를 압도하는 것을 보여주었습니다 데이터들을 주지 않고 그저 처음 보는 게임을 시켜보았을 뿐인데, 시간에 지남에 따라 스스로 학습하고 가장 효율적인 플레이를 하게 되는 것에 큰 흥미를 느끼게 되어 프로젝트 주제로 선정하게 되었습니다.<br>

다루게 될 기술
-------
1. LuaJIT
2. Torch 7.0
3. nngraph<br>

프로젝트 개요
-------
1. 게임 데이터: Breakout (Atari)
2. 알고리즘: DQN Breakout Algorithm
3. 출처: https://github.com/deepmind/dqn
4. Atari사의 벽돌깨기 게임인 ‘Breakout’과 DeepMind사의 DQN 알고리즘을 통해 인공지능에
게 게임을 학습시킨다.<br>

동영상 링크
-----


기대효과
-------
1. 데이터를 따로 주지 않았음에도, 픽셀과 보상(게임 점수)이라는 두 가지 만으로 스스로 게임의 진행과 플레이 방식에 대해 알아내는 것을 확인할 수 있음. 
2. 게임이 진행될수록 인공지능이 발전하여 그저 플레이 하는 것에서 더욱 나아가 한층 발전된 전략 등을 구사하는 것으로 AI의 잠재력과 가능성을 시사함.<br>


DQN Agent Code
------
```python
# 브레이크아웃에서의 DQN 에이전트
class DQNAgent:
    def __init__(self, action_size):
        self.render = False
        self.load_model = False
        # 상태와 행동의 크기 정의
        self.state_size = (84, 84, 4)
        self.action_size = action_size
        # DQN 하이퍼파라미터
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 1000000.
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                  / self.exploration_steps
        self.batch_size = 32
        self.train_start = 50000
        self.update_target_rate = 10000
        self.discount_factor = 0.99
        # 리플레이 메모리, 최대 크기 400000
        self.memory = deque(maxlen=400000)
        self.no_op_steps = 30
        # 모델과 타겟모델을 생성하고 타겟모델 초기화
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
 
        self.optimizer = self.optimizer()
 
        # 텐서보드 설정
        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)
 
        self.avg_q_max, self.avg_loss = 0, 0
        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            'summary/breakout_dqn', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
 
        if self.load_model:
            self.model.load_weights("./save_model/breakout_dqn_trained.h5")
 
    # Huber Loss를 이용하기 위해 최적화 함수를 직접 정의
    def optimizer(self):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')
 
        prediction = self.model.output
 
        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(prediction * a_one_hot, axis=1)
        error = K.abs(y - q_value)
 
        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
 
        optimizer = RMSprop(lr=0.00025, epsilon=0.01)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, a, y], [loss], updates=updates)
 
        return train
 
    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4),
                         activation='relu',
                         input_shape=self.state_size))
        model.add(Conv2D(64, (4, 4), strides=(2, 2),
                         activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1),
                         activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size))
        model.summary()
        return model
 
    # 타겟 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
 
    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])
 
    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, history, action, reward, next_history, dead):
        self.memory.append((history, action, reward, next_history, dead))
 
    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step
 
        mini_batch = random.sample(self.memory, self.batch_size)
 
        history = np.zeros((self.batch_size, self.state_size[0],
                            self.state_size[1], self.state_size[2]))
        next_history = np.zeros((self.batch_size, self.state_size[0],
                                 self.state_size[1], self.state_size[2]))
        target = np.zeros((self.batch_size,))
        action, reward, dead = [], [], []
 
        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0] / 255.)
            next_history[i] = np.float32(mini_batch[i][3] / 255.)
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            dead.append(mini_batch[i][4])
 
        target_value = self.target_model.predict(next_history)
 
        for i in range(self.batch_size):
            if dead[i]:
                target[i] = reward[i]
            else:
                target[i] = reward[i] + self.discount_factor * \
                                        np.amax(target_value[i])
 
        loss = self.optimizer([history, action, target])
        self.avg_loss += loss[0]
 
    # 각 에피소드 당 학습 정보를 기록
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)
 
        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)
        tf.summary.scalar('Average Loss/Episode', episode_avg_loss)
 
        summary_vars = [episode_total_reward, episode_avg_max_q,
                        episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op
```
<br>
Main Code
----------

```python
if __name__ == "__main__":
    # 환경과 DQN 에이전트 생성
    env = gym.make('BreakoutDeterministic-v4')
    agent = DQNAgent(action_size=3)
 
    scores, episodes, global_step = [], [], 0
 
    for e in range(EPISODES):
        done = False
        dead = False
 
        step, score, start_life = 0, 0, 5
        observe = env.reset()
 
        for _ in range(random.randint(1, agent.no_op_steps)):
            observe, _, _, _ = env.step(1)
 
        state = pre_processing(observe)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))
 
        while not done:
            if agent.render:
                env.render()
            global_step += 1
            step += 1
 
            # 바로 전 4개의 상태로 행동을 선택
            action = agent.get_action(history)
            # 1: 정지, 2: 왼쪽, 3: 오른쪽
            if action == 0:
                real_action = 1
            elif action == 1:
                real_action = 2
            else:
                real_action = 3
 
            # 선택한 행동으로 환경에서 한 타임스텝 진행
            observe, reward, done, info = env.step(real_action)
            # 각 타임스텝마다 상태 전처리
            next_state = pre_processing(observe)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)
 
            agent.avg_q_max += np.amax(
                agent.model.predict(np.float32(history / 255.))[0])
 
            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']
 
            reward = np.clip(reward, -1., 1.)
            # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장 후 학습
            agent.append_sample(history, action, reward, next_history, dead)
 
            if len(agent.memory) >= agent.train_start:
                agent.train_model()
 
            # 일정 시간마다 타겟모델을 모델의 가중치로 업데이트
            if global_step % agent.update_target_rate == 0:
                agent.update_target_model()
 
            score += reward
 
            if dead:
                dead = False
            else:
                history = next_history
 
            if done:
                # 각 에피소드 당 학습 정보를 기록
                if global_step > agent.train_start:
                    stats = [score, agent.avg_q_max / float(step), step,
                             agent.avg_loss / float(step)]
 
                    for i in range(len(stats)):
                        agent.sess.run(agent.update_ops[i], feed_dict={
                            agent.summary_placeholders[i]: float(stats[i])
                        })
                    summary_str = agent.sess.run(agent.summary_op)
                    agent.summary_writer.add_summary(summary_str, e + 1)
 
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon,
                      "  global_step:", global_step, "  average_q:",
                      agent.avg_q_max / float(step), "  average loss:",
                      agent.avg_loss / float(step))
 
                agent.avg_q_max, agent.avg_loss = 0, 0
 
        # 1000 에피소드마다 모델 저장
        if e % 1000 == 0:
            agent.model.save_weights("./save_model/breakout_dqn.h5")
```
<br>
<h3>학습시키는 법</h3><br>
1. 리눅스(Ubuntu) 개발 환경을 구축한다. (18.04.5 이하 추천)<br>
2. 아래 순서로 명령어들을 입력하여 torch를 설치한다.<br>

`$ git clone https://github.com/torch/distro.git ~/torch --recursive`<br>
`$ cd ~/torch`<br>
`$ bash install-deps;`<br>
`$ ./install.sh`<br>

3. 아래 순서로 명령어들을 입력하여 학습을 시작한다.<br>
clone DQN algorithm in github
`$ git clone https://github.com/deepmind/dqn`<br>
`$ ./install_dependencies.sh`<br>
roms 폴더에 breakout.bin 파일을 다운받은 뒤<br>
`$ ./run_cpu breakout`<br>

한계
---------
1. 학습 시간이 매우 오래걸린다.
2. 최근에 나온 알고리즘이 아니라서 deprecated 된 함수가 많아, 게임을 학습하는 것을 display 하지 못하였다.
