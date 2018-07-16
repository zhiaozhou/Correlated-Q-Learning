from env import *
from cvxopt import matrix, solvers

gamma = .9 # discount
alpha = .1 # learning rate
e = .2

s_test = 71 #[1,2,1]
a_test = 21 #[4,0]

n_iter = 1000  # numer of iterations of Q learning
timeout = 25  # episode length timeout threshold, shoud never occur

pi1 = np.ones((112,5))/5
pi2 = np.ones((112,5))/5

e_decayrate = e/(10*n_iter)
alpha_decayrate = alpha/n_iter

# initialize Q to zero over the states and joint action space
Q1 = np.random.rand(112, 5, 5)
Q2 = np.random.rand(112, 5, 5)
#
# V1 = np.ones(112,5)
# V2 = np.ones(112,5)

rewards = []

s0 = 71 # Alwats start at the same position, as in the pic
ERR = [] # delta in Q(s,a)
prob = .2 # 1/len(actions)

def findMinQ2(Q1):

    c = matrix([0.,0.,0.,0.,0.,1.0])

    G = matrix(np.array(
        [
    [-1.0, 0., 0., 0., 0., 0.],
    [ 0., -1.0, 0., 0., 0., 0.],
    [ 0., 0., -1.0, 0., 0., 0.],
    [ 0., 0., 0., -1.0, 0., 0.],
    [ 0., 0., 0., 0., -1.0, 0.],
    [ 0., 0., 0., 0., 0., -1.0],
    [Q1[0, 0], Q1[0, 1], Q1[0, 2], Q1[0, 3], Q1[0, 4], -1.],
    [Q1[1, 0], Q1[1, 1], Q1[1, 2], Q1[1, 3], Q1[1, 4], -1.],
    [Q1[2, 0], Q1[2, 1], Q1[2, 2], Q1[2, 3], Q1[2, 4], -1.],
    [Q1[3, 0], Q1[3, 1], Q1[3, 2], Q1[3, 3], Q1[3, 4], -1.],
    [Q1[4, 0], Q1[4, 1], Q1[4, 2], Q1[4, 3], Q1[4, 4], -1.]
    ])
    )
    h = matrix([ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    A = matrix([
        [1.],[1.],[1.],[1.],[1.],[0.]
    ])

    b = matrix([1.])

    solvers.options['show_progress'] = False
    sol = solvers.lp(c, G, h, A, b)

    primeDist = [sol['x'][0],sol['x'][1],sol['x'][2],sol['x'][3],sol['x'][4]]
    return primeDist

def findMinQ(Qmin):

    c = matrix([
        # Qmin[0, 0] + Qmin[1, 0] + Qmin[2, 0] + Qmin[3, 0] + Qmin[4, 0],
        # Qmin[0, 1] + Qmin[1, 1] + Qmin[2, 1] + Qmin[3, 1] + Qmin[4, 1],
        # Qmin[0, 2] + Qmin[1, 2] + Qmin[2, 2] + Qmin[3, 2] + Qmin[4, 2],
        # Qmin[0, 3] + Qmin[1, 3] + Qmin[2, 3] + Qmin[3, 3] + Qmin[4, 3],
        # Qmin[0, 4] + Qmin[1, 4] + Qmin[2, 4] + Qmin[3, 4] + Qmin[4, 4]

        Qmin[0, 0] + Qmin[0, 1] + Qmin[0, 2] + Qmin[0, 3] + Qmin[0, 4],
        Qmin[1, 0] + Qmin[1, 1] + Qmin[1, 2] + Qmin[1, 3] + Qmin[1, 4],
        Qmin[2, 0] + Qmin[2, 1] + Qmin[2, 2] + Qmin[2, 3] + Qmin[2, 4],
        Qmin[3, 0] + Qmin[3, 1] + Qmin[3, 2] + Qmin[3, 3] + Qmin[3, 4],
        Qmin[4, 0] + Qmin[4, 1] + Qmin[4, 2] + Qmin[4, 3] + Qmin[4, 4]
    ])

    G = matrix([
    [-1.0, 0., 0., 0., 0.],
    [ 0., -1.0, 0., 0., 0.],
    [ 0., 0., -1.0, 0., 0.],
    [ 0., 0., 0., -1.0, 0.],
    [ 0., 0., 0., 0., -1.0]
    ])
    h = matrix([ 0.0, 0.0, 0.0, 0.0, 0.0])

    A = matrix([
        [1.],[1.],[1.],[1.],[1.]
    ])

    b = matrix([1.])
    solvers.options['show_progress'] = False
    sol = solvers.lp(c, G, h, A, b)

    primeDist = [sol['x'][0],sol['x'][1],sol['x'][2],sol['x'][3],sol['x'][4]]
    return primeDist

for T in range(n_iter):
    s = s0  # always initalize an episode in s0
    q_sa = Q1[s_test, 4, 0]

    for t in range(timeout):
        choice = np.random.rand()

        if choice <= e:
            a1 = actions[np.random.randint(5)]
            a2 = actions[np.random.randint(5)]
        else:
            pi1[s] = np.array(findMinQ(Q1[s])) # find min value of dist for me
            # a1 = actions[np.random.choice(primeDista2)]
            pi2[s]= np.array(findMinQ(Q2[s])) # find min value of dist for me

            # a1 = np.random.choice(actions, p = primeDista2)
            # a2 = np.random.choice(actions, p = primeDista1)

            a1 = np.argmax(pi1[s])
            a2 = np.argmax(pi2[s])

        a = [a1, a2] # action matrix
        # print a

        s_prime = transition(s, a) # query transition model to obtain s', returns an index value

        # query the reward model to obtain r
        r1 = R[s_prime, 0]
        r2 = R[s_prime, 1]

        # Q1[s, a1, a2] = (1. - alpha) * Q1[s, a1, a2] + alpha * (r1 + gamma * np.min(Q1[s, :, :].sum(axis=0) * primeDista2))
        # Q2[s, a2, a1] = (1. - alpha) * Q2[s, a2, a1] + alpha * (r2 + gamma * np.min(Q2[s, :, :].sum(axis=0) * primeDista1))

        v1 = pi1[s_prime]* Q1[s_prime]
        v2 = pi2[s_prime]* Q2[s_prime]
        Q1[s, a1, a2] = (1. - alpha) * Q1[s, a1, a2] + alpha * (r1 + gamma * v1.max())
        Q2[s, a2, a1] = (1. - alpha) * Q2[s, a2, a1] + alpha * (r2 + gamma * v2.max())

        # Q1[s, a1, a2min] = (1.-alpha)*Q1[s, a1, a2min] + alpha * (r1 + gamma * Q1[s_prime, :].max())
        # Q2[s, a2, a1min] = (1.-alpha)*Q2[s, a2, a1min] + alpha * (r2 + gamma * Q2[s_prime, :].max())
        # Q1[s, a1] = Q1[s, a1] + alpha * (r1 + gamma * Q1[s_prime, :].max() - Q1[s, a1])
        # Q2[s, a2] = Q2[s, a2] + alpha * (r2 + gamma * Q2[s_prime, :].max() - Q2[s, a2])
        # update s
        s = s_prime
        if e > .001:
            e = e - e_decayrate

        # terminate when a goal is made
        if r1 != 0 or r2 != 0:
            rewards.append([r1, r2])
            break

    # if alpha > .001:
    #     alpha = alpha - alpha_decayrate

    ERR.append(np.abs(Q1[s_test, 4, 0] - q_sa))
    # print Q1[0]
    print T


# print Q1

plt.plot(ERR)
plt.ylim([0,.5])
plt.show()
