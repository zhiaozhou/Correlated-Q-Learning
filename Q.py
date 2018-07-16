from env import *

e = .5 # epsilon
gamma = .99 # discount
alpha = .5 # learning rate

n_iter = 10000  # numer of iterations of Q learning
timeout = 25  # episode length timeout threshold, shoud never occur

e_decayrate = (e-.001)/n_iter
alpha_decayrate = (alpha-.001)/n_iter

# initialize Q to zero over the states and joint action space
Q1 = np.random.rand(len(S), len(actions))
Q2 = np.random.rand(len(S), len(actions))

s0 = 71 # Alwats start at the same position, as in the pic

ERR = [] # delta in Q(s,a)

for T in range(n_iter):
    s = s0  # always initalize an episode in s0
    q_sa = Q1[s0, 4]

    for t in range(timeout):
        # epsilon-greedily select an action
        # a1 = e_greedy(Q1, epsilon(T))
        choice = np.random.rand()
        if choice <= e:
            a1 = actions[np.random.randint(5)]
            a2 = actions[np.random.randint(5)]
        else:

            a1 = actions[np.argmax(Q1[s])] #max(Q1[s]).astype(int) -4, 1, 0
            a2 = actions[np.argmax(Q2[s])]

        a = [a1, a2] # action matrix
        s_prime = transition(s, a) # query transition model to obtain s', returns an index value

        # query the reward model to obtain r
        r1 = R[s_prime, 0]
        r2 = R[s_prime, 1]

        # update Q
        # Q1[s,a1] = (1 - alpha) * Q1[s,a1] + alpha * ((1. - gamma) * r1 + gamma * Q1[s_prime,:].max())
        # Q2[s,a2] = (1 - alpha) * Q2[s,a2] + alpha * ((1. - gamma) * r2 + gamma * Q2[s_prime,:].max())

        Q1[s, a1] = Q1[s, a1] + alpha * (r1 + gamma * Q1[s_prime, :].max() - Q1[s, a1])
        Q2[s, a2] = Q2[s, a2] + alpha * (r2 + gamma * Q2[s_prime, :].max() - Q2[s, a2])
        # update s
        s = s_prime



        # terminate when a goal is made
        if r1 != 0 or r2 != 0: break
    # alpha = alpha * .999
    # Decay Alpha
    e = e - e_decayrate
    # if e > .001:
    #     e = e - e_decayrate

    # alpha = alpha*.999
    if alpha > .001:
        alpha = alpha - alpha_decayrate
    ERR.append(np.abs(Q1[s0, 4] - q_sa))
    # print np.abs(Q1[s0, 4] - q_sa)
    print T

# for i in range(len(ERR)):
#     print ERR[i]
# # print ERR
#

print Q1

plt.plot(ERR)
plt.ylim([0,.5])
plt.show()