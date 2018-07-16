from env import *
from cvxopt import matrix, solvers
#
# A = np.array([
#        [-4, -4],
#        [ 4, -4],
#        [ 1, -4],
#        [-1, -4],
#        [ 0, -4],
#        [-4,  4],
#        [ 4,  4],
#        [ 1,  4],
#        [-1,  4],
#        [ 0,  4],
#        [-4,  1],
#        [ 4,  1],
#        [ 1,  1],
#        [-1,  1],
#        [ 0,  1],
#        [-4, -1],
#        [ 4, -1],
#        [ 1, -1],
#        [-1, -1],
#        [ 0, -1],
#        [-4,  0],
#        [ 4,  0],
#        [ 1,  0],
#        [-1,  0],
#        [ 0,  0]])

# gamma = .9 # discount
# alpha = .1 # learning rate
# e = .9
#
# s_test = 71 #[1,2,1]
# a_test = 21 #[4,0]
#
# n_iter = 5000  # numer of iterations of Q learning
# timeout = 25  # episode length timeout threshold, shoud never occur
#
# e_decay = 1/n_iter


gamma = .9 # discount
alpha = .5 # learning rate
e = .9

s_test = 71 #[1,2,1]
a_test = 21 #[4,0]

n_iter = 5000  # numer of iterations of Q learning
timeout = 25  # episode length timeout threshold, shoud never occur

# pi1 = np.ones((112,5))/5
# pi2 = np.ones((112,5))/5

e_decayrate = e/(n_iter)
alpha_decayrate = alpha/n_iter

# initialize Q to zero over the states and joint action space
Q1 = np.random.rand(112, 25)
Q2 = np.random.rand(112, 25)

pi1 = np.ones((112,25))
pi2 = np.ones((112,25))


s0 = 71 # Alwats start at the same position, as in the pic

ERR = [] # delta in Q(s,a)

prob = .9 # 1/len(actions)

def findMinQ2(Q1):

    c = matrix(
    [
        -Q1[0,0],
        -Q1[0, 1],
        -Q1[0, 2],
        -Q1[0, 3],
        -Q1[0, 4],
        -Q1[1, 0],
        -Q1[1, 1],
        -Q1[1, 2],
        -Q1[1, 3],
        -Q1[1, 4],
        -Q1[2, 0],
        -Q1[2, 1],
        -Q1[2, 2],
        -Q1[2, 3],
        -Q1[2, 4],
        -Q1[3, 0],
        -Q1[3, 1],
        -Q1[3, 2],
        -Q1[3, 3],
        -Q1[3, 4],
        -Q1[3, 0],
        -Q1[4, 1],
        -Q1[4, 2],
        -Q1[4, 3],
        -Q1[4, 4],

        -Q2[0, 0],
        -Q2[0, 1],
        -Q2[0, 2],
        -Q2[0, 3],
        -Q2[0, 4],
        -Q2[1, 0],
        -Q2[1, 1],
        -Q2[1, 2],
        -Q2[1, 3],
        -Q2[1, 4],
        -Q2[2, 0],
        -Q2[2, 1],
        -Q2[2, 2],
        -Q2[2, 3],
        -Q2[2, 4],
        -Q2[3, 0],
        -Q2[3, 1],
        -Q2[3, 2],
        -Q2[3, 3],
        -Q2[3, 4],
        -Q2[3, 0],
        -Q2[4, 1],
        -Q2[4, 2],
        -Q2[4, 3],
        -Q2[4, 4]
        ]
    )

    # c = matrix([
    #     -Q1[0],-Q1[1],-Q1[2],-Q1[3],-Q1[4],-Q1[5],-Q1[6],-Q1[7],-Q1[8],-Q1[9],
    #     -Q1[10],-Q1[11],-Q1[12],-Q1[13],-Q1[14],-Q1[15],-Q1[16],-Q1[17],-Q1[18],-Q1[19],
    #     -Q1[20],-Q1[21],-Q1[22],-Q1[23],-Q1[24]
    #     ])

    G = matrix(np.identity(50)*-1)

    h = matrix([ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0])

    A = matrix([
        [1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],[1.],
        [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.],
        [1.], [1.], [1.], [1.], [1.],
        [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.],
        [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.],
        [1.], [1.], [1.], [1.], [1.]
    ])

    b = matrix([1.])
    solvers.options['show_progress'] = False
    sol = solvers.lp(c, G, h, A, b)

    primeDist = [sol['x'][0],sol['x'][1],sol['x'][2],sol['x'][3],sol['x'][4],sol['x'][5],sol['x'][6],sol['x'][7],sol['x'][8],sol['x'][9],
                 sol['x'][10], sol['x'][11], sol['x'][12], sol['x'][13], sol['x'][14], sol['x'][15], sol['x'][16], sol['x'][17], sol['x'][18], sol['x'][19],
                 sol['x'][20], sol['x'][21], sol['x'][22], sol['x'][23], sol['x'][24],
                 #
                 # sol['x'][25], sol['x'][26], sol['x'][27], sol['x'][28], sol['x'][29], sol['x'][30], sol['x'][31], sol['x'][32],
                 # sol['x'][33], sol['x'][34],
                 # sol['x'][35], sol['x'][36], sol['x'][37], sol['x'][38], sol['x'][39], sol['x'][40], sol['x'][41],
                 # sol['x'][42], sol['x'][43], sol['x'][44],
                 # sol['x'][45], sol['x'][46], sol['x'][47], sol['x'][48], sol['x'][49]

                 ]
    return primeDist

def findMinQ(Q1, Q2):
    # c = matrix(
    #     [
    #         # -Q1[0, 0],
    #         # -Q1[0, 1],
    #         # -Q1[0, 2],
    #         # -Q1[0, 3],
    #         # -Q1[0, 4],
    #         # -Q1[1, 0],
    #         # -Q1[1, 1],
    #         # -Q1[1, 2],
    #         # -Q1[1, 3],
    #         # -Q1[1, 4],
    #         # -Q1[2, 0],
    #         # -Q1[2, 1],
    #         # -Q1[2, 2],
    #         # -Q1[2, 3],
    #         # -Q1[2, 4],
    #         # -Q1[3, 0],
    #         # -Q1[3, 1],
    #         # -Q1[3, 2],
    #         # -Q1[3, 3],
    #         # -Q1[3, 4],
    #         # -Q1[3, 0],
    #         # -Q1[4, 1],
    #         # -Q1[4, 2],
    #         # -Q1[4, 3],
    #         # -Q1[4, 4],
    #         #
    #         # -Q2[0, 0],
    #         # -Q2[0, 1],
    #         # -Q2[0, 2],
    #         # -Q2[0, 3],
    #         # -Q2[0, 4],
    #         # -Q2[1, 0],
    #         # -Q2[1, 1],
    #         # -Q2[1, 2],
    #         # -Q2[1, 3],
    #         # -Q2[1, 4],
    #         # -Q2[2, 0],
    #         # -Q2[2, 1],
    #         # -Q2[2, 2],
    #         # -Q2[2, 3],
    #         # -Q2[2, 4],
    #         # -Q2[3, 0],
    #         # -Q2[3, 1],
    #         # -Q2[3, 2],
    #         # -Q2[3, 3],
    #         # -Q2[3, 4],
    #         # -Q2[3, 0],
    #         # -Q2[4, 1],
    #         # -Q2[4, 2],
    #         # -Q2[4, 3],
    #         # -Q2[4, 4]
    #     ]
    # )

    c = matrix([
        -Q1[0],-Q1[1],-Q1[2],-Q1[3],-Q1[4],-Q1[5],-Q1[6],-Q1[7],-Q1[8],-Q1[9],
        -Q1[10],-Q1[11],-Q1[12],-Q1[13],-Q1[14],-Q1[15],-Q1[16],-Q1[17],-Q1[18],-Q1[19],
        -Q1[20],-Q1[21],-Q1[22],-Q1[23],-Q1[24],

        -Q2[0], -Q2[1], -Q2[2], -Q2[3], -Q2[4], -Q2[5], -Q2[6], -Q2[7], -Q2[8], -Q2[9],
        -Q2[10], -Q2[11], -Q2[12], -Q2[13], -Q2[14], -Q2[15], -Q2[16], -Q2[17], -Q2[18], -Q2[19],
        -Q2[20], -Q2[21], -Q2[22], -Q2[23], -Q2[24],
        ])

    G = matrix(np.identity(50) * -1)

    h = matrix(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0])

    A = matrix([
        [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.],
        [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.],
        [1.], [1.], [1.], [1.], [1.],
        [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.],
        [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.], [1.],
        [1.], [1.], [1.], [1.], [1.]
    ])

    b = matrix([1.])
    solvers.options['show_progress'] = False
    sol = solvers.lp(c, G, h, A, b)

    primeDist = [sol['x'][0], sol['x'][1], sol['x'][2], sol['x'][3], sol['x'][4], sol['x'][5], sol['x'][6], sol['x'][7],
                 sol['x'][8], sol['x'][9],
                 sol['x'][10], sol['x'][11], sol['x'][12], sol['x'][13], sol['x'][14], sol['x'][15], sol['x'][16],
                 sol['x'][17], sol['x'][18], sol['x'][19],
                 sol['x'][20], sol['x'][21], sol['x'][22], sol['x'][23], sol['x'][24],

                 # sol['x'][25], sol['x'][26], sol['x'][27], sol['x'][28], sol['x'][29], sol['x'][30], sol['x'][31],
                 # sol['x'][32],
                 # sol['x'][33], sol['x'][34],
                 # sol['x'][35], sol['x'][36], sol['x'][37], sol['x'][38], sol['x'][39], sol['x'][40], sol['x'][41],
                 # sol['x'][42], sol['x'][43], sol['x'][44],
                 # sol['x'][45], sol['x'][46], sol['x'][47], sol['x'][48], sol['x'][49]

                 ]
    return primeDist


for T in range(n_iter):
    s = s0  # always initalize an episode in s0
    q_sa = Q1[s_test, 21]

    for t in range(timeout):
        choice = np.random.rand()
        # QtableV = Q1.sum()

        if choice <= e:
            a1 = np.random.randint(25)
            a2 = np.random.randint(25)
        else:

            prime1 = findMinQ(Q1[s], Q2[s]) # return an array of 25 elements with probs for A1 only
            prime2 = findMinQ(Q2[s], Q1[s])

            pi1[s] = prime1 # ideally, I would want this to be averaged
            pi2[s] = prime2

            QwithProb1 = Q1[s]*prime1 # 25 prob valuea multiplied by anpther 25 Q values
            QwithProb2 = Q2[s]*prime2

            a1c = np.argmax(QwithProb1) # returns a number index where it is
            a2c = np.argmax(QwithProb2)

            # primeDista1 = np.array(findMinQ(Q1[s])) # find min value of dist for me
            # primeDista2 = np.array(findMinQ(Q2[s]))  # find min value of dist for me
            # a1c = np.argmax(primeDista1)
            # a2c = np.argmax(primeDista2)

            a1 = A[a1c][0]
            a2 = A[a2c][0]

            # finalPrime = primeDista1+primeDista2
            #
            # print primeDista1
            # print primeDista2
            # print finalPrime
            #
            # aCn = np.argmax(finalPrime)
            # aC = A[aCn]
            # a1 = aC[0]
            # a2 = aC[1]
            # a1 = a1c[0]
            #
            # # primeDista2 = np.array(findMinQ(Q2, s))  # find min value of dist for me
            #
            # a2 = a2c[1]

            # old
            # a1min = np.argmin(primeDista1)  # will return position of an action that minimizes me
            # a2c = np.argmax(Q1[s, :, a1min])
            # a2 = actions[a2c]
            #
            # a2c = np.argmax(Q2[s])
            # a2 = A[a2c][1]
        # finalPrime = primeDista1 + primeDista2
        # abothC = np.argmax(finalPrime)
        # a1 = A[abothC][0]
        # a2 = A[abothC][1]
        # a1 = A[a1c][0]
        # a2 = A[a2c][1]
        # findMinQ(Q1, s)

        a = [a1, a2] # action matrix

        s_prime = transition(s, a) # query transition model to obtain s', returns an index value

        # query the reward model to obtain r
        r1 = R[s_prime, 0]
        r2 = R[s_prime, 1]

        anp = np.array(a)
        for i in range(A.shape[0]):
            if np.array_equal(A[i], anp):
                aindex = i
                break
        anp = np.array([a2,a1])
        for j in range(A.shape[0]):
            if np.array_equal(A[j], anp):
                aindex = j
                break

        v1 = Q1[s_prime]*pi1[s_prime]
        v2 = Q2[s_prime]*pi2[s_prime]

        Q1[s, i] = (1. - alpha) * Q1[s, i] + alpha * ((1-gamma)*r1 + gamma * v1.max())
        Q2[s, j] = (1. - alpha) * Q2[s, j] + alpha * ((1-gamma)*r2 + gamma * v2.max())

        # Q1[s, a1, a2] = (1. - alpha) * Q1[s, a1, a2] + alpha * (r1 + gamma * Q1[s_prime].max())
        # Q2[s, a2, a1] = (1. - alpha) * Q2[s, a2, a1] + alpha * (r2 + gamma * Q2[s_prime].max())

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
            # rewards.append([r1, r2])
            break

            # if alpha > .001:
            #     alpha = alpha - alpha_decayrate

    ERR.append(np.abs(Q1[s_test, 21] - q_sa))
    # print Q1[0]
    print T
    #     s = s_prime
    #
    #     # QtableV2 = Q1.sum()
    #     # print QtableV2-QtableV
    #     # alpha = alpha*.999
    #
    #
    #     # terminate when a goal is made
    #     if r1 != 0 or r2 != 0:
    #         break
    #
    # if e > .001:
    #     e = e - e_decay
    if alpha > .001:
        alpha = alpha * .99999
    # #
    # ERR.append(np.abs(Q1[s_test, 21] - q_sa))
    # # print Q1[0]
    # print T

# print 'results:'
# for i in range(len(ERR)):
#     print ERR[i]


# print Q1

plt.plot(ERR)
plt.ylim([0,.5])
plt.show()
