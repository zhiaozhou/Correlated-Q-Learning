import numpy as np
import matplotlib.pyplot as plt



b = 2 # ball, A or B has it
s = 8 # number of states

# # ACTIONS:
# N, S, E, W, P
actions = [-4,4,1,-1,0]
A = np.array([[-4, -4],
       [ 4, -4],
       [ 1, -4],
       [-1, -4],
       [ 0, -4],
       [-4,  4],
       [ 4,  4],
       [ 1,  4],
       [-1,  4],
       [ 0,  4],
       [-4,  1],
       [ 4,  1],
       [ 1,  1],
       [-1,  1],
       [ 0,  1],
       [-4, -1],
       [ 4, -1],
       [ 1, -1],
       [-1, -1],
       [ 0, -1],
       [-4,  0],
       [ 4,  0],
       [ 1,  0],
       [-1,  0],
       [ 0,  0]])
S = np.array([[0, 0, 1],
       [0, 0, 2],
       [0, 0, 3],
       [0, 0, 4],
       [0, 0, 5],
       [0, 0, 6],
       [0, 0, 7],
       [0, 1, 0],
       [0, 1, 2],
       [0, 1, 3],
       [0, 1, 4],
       [0, 1, 5],
       [0, 1, 6],
       [0, 1, 7],
       [0, 2, 0],
       [0, 2, 1],
       [0, 2, 3],
       [0, 2, 4],
       [0, 2, 5],
       [0, 2, 6],
       [0, 2, 7],
       [0, 3, 0],
       [0, 3, 1],
       [0, 3, 2],
       [0, 3, 4],
       [0, 3, 5],
       [0, 3, 6],
       [0, 3, 7],
       [0, 4, 0],
       [0, 4, 1],
       [0, 4, 2],
       [0, 4, 3],
       [0, 4, 5],
       [0, 4, 6],
       [0, 4, 7],
       [0, 5, 0],
       [0, 5, 1],
       [0, 5, 2],
       [0, 5, 3],
       [0, 5, 4],
       [0, 5, 6],
       [0, 5, 7],
       [0, 6, 0],
       [0, 6, 1],
       [0, 6, 2],
       [0, 6, 3],
       [0, 6, 4],
       [0, 6, 5],
       [0, 6, 7],
       [0, 7, 0],
       [0, 7, 1],
       [0, 7, 2],
       [0, 7, 3],
       [0, 7, 4],
       [0, 7, 5],
       [0, 7, 6],
       [1, 0, 1],
       [1, 0, 2],
       [1, 0, 3],
       [1, 0, 4],
       [1, 0, 5],
       [1, 0, 6],
       [1, 0, 7],
       [1, 1, 0],
       [1, 1, 2],
       [1, 1, 3],
       [1, 1, 4],
       [1, 1, 5],
       [1, 1, 6],
       [1, 1, 7],
       [1, 2, 0],
       [1, 2, 1],
       [1, 2, 3],
       [1, 2, 4],
       [1, 2, 5],
       [1, 2, 6],
       [1, 2, 7],
       [1, 3, 0],
       [1, 3, 1],
       [1, 3, 2],
       [1, 3, 4],
       [1, 3, 5],
       [1, 3, 6],
       [1, 3, 7],
       [1, 4, 0],
       [1, 4, 1],
       [1, 4, 2],
       [1, 4, 3],
       [1, 4, 5],
       [1, 4, 6],
       [1, 4, 7],
       [1, 5, 0],
       [1, 5, 1],
       [1, 5, 2],
       [1, 5, 3],
       [1, 5, 4],
       [1, 5, 6],
       [1, 5, 7],
       [1, 6, 0],
       [1, 6, 1],
       [1, 6, 2],
       [1, 6, 3],
       [1, 6, 4],
       [1, 6, 5],
       [1, 6, 7],
       [1, 7, 0],
       [1, 7, 1],
       [1, 7, 2],
       [1, 7, 3],
       [1, 7, 4],
       [1, 7, 5],
       [1, 7, 6]])
R = np.array([[ 100., -100.],
       [ 100., -100.],
       [ 100., -100.],
       [ 100., -100.],
       [ 100., -100.],
       [ 100., -100.],
       [ 100., -100.],
       [   0.,    0.],
       [   0.,    0.],
       [   0.,    0.],
       [   0.,    0.],
       [   0.,    0.],
       [   0.,    0.],
       [   0.,    0.],
       [   0.,    0.],
       [   0.,    0.],
       [   0.,    0.],
       [   0.,    0.],
       [   0.,    0.],
       [   0.,    0.],
       [   0.,    0.],
       [-100.,  100.],
       [-100.,  100.],
       [-100.,  100.],
       [-100.,  100.],
       [-100.,  100.],
       [-100.,  100.],
       [-100.,  100.],
       [ 100., -100.],
       [ 100., -100.],
       [ 100., -100.],
       [ 100., -100.],
       [ 100., -100.],
       [ 100., -100.],
       [ 100., -100.],
       [   0.,    0.],
       [   0.,    0.],
       [   0.,    0.],
       [   0.,    0.],
       [   0.,    0.],
       [   0.,    0.],
       [   0.,    0.],
       [   0.,    0.],
       [   0.,    0.],
       [   0.,    0.],
       [   0.,    0.],
       [   0.,    0.],
       [   0.,    0.],
       [   0.,    0.],
       [-100.,  100.],
       [-100.,  100.],
       [-100.,  100.],
       [-100.,  100.],
       [-100.,  100.],
       [-100.,  100.],
       [-100.,  100.],
       [   0.,    0.],
       [   0.,    0.],
       [-100.,  100.],
       [ 100., -100.],
       [   0.,    0.],
       [   0.,    0.],
       [-100.,  100.],
       [ 100., -100.],
       [   0.,    0.],
       [-100.,  100.],
       [ 100., -100.],
       [   0.,    0.],
       [   0.,    0.],
       [-100.,  100.],
       [ 100., -100.],
       [   0.,    0.],
       [-100.,  100.],
       [ 100., -100.],
       [   0.,    0.],
       [   0.,    0.],
       [-100.,  100.],
       [ 100., -100.],
       [   0.,    0.],
       [   0.,    0.],
       [ 100., -100.],
       [   0.,    0.],
       [   0.,    0.],
       [-100.,  100.],
       [ 100., -100.],
       [   0.,    0.],
       [   0.,    0.],
       [-100.,  100.],
       [   0.,    0.],
       [   0.,    0.],
       [-100.,  100.],
       [ 100., -100.],
       [   0.,    0.],
       [   0.,    0.],
       [-100.,  100.],
       [ 100., -100.],
       [   0.,    0.],
       [-100.,  100.],
       [ 100., -100.],
       [   0.,    0.],
       [   0.,    0.],
       [-100.,  100.],
       [ 100., -100.],
       [   0.,    0.],
       [-100.,  100.],
       [ 100., -100.],
       [   0.,    0.],
       [   0.,    0.],
       [-100.,  100.],
       [ 100., -100.],
       [   0.,    0.],
       [   0.,    0.]])

def transition(s, a):
    ball, me, you = S[s]

    # a1 = actions[a[0]]
    # a2 = actions[a[1]]
    #
    # a = [a1, a2] # THIS IS DUMB... FIX IT

    # WE CANNOT GO OVERBOARD!!!
    if (me + a[0] > 7 or me + a[0] < 0):
        a[0] = 0
    if (you + a[1] > 7 or you + a[1] < 0):
        a[1] = 0

    mynewstate = me + a[0]
    yournewstate = me + a[1]

    s_ind = s

    # GETTING NEW STATE
    if mynewstate == yournewstate:
        first = np.random.randint(2)
        if first == 0:  # I'm first
            if ball == 0:  # I have a ball
                s_prime = [ball, me + a[0], you]  # I move, you don't, ball is still mine
            else:
                s_prime = [ball - 1, me + a[0], you]  # i move, you bump into me, I get your ball
        else:  # you move first
            if ball == 1:  # you have the ball
                s_prime = [ball, me, you + a[1]]  # you move, you keep the ball, i stay
            else:  # i have the ball
                s_prime = [ball - 1, me, you + a[1]]  # you move, I bump into you, you get the ball
    else:
        s_prime = [ball, me + a[0], you + a[1]]

    np_s_prime = np.array(s_prime)
    # GET INDEX OF S' STATE
    for i in range(S.shape[0]):
        if np.array_equal(S[i], np_s_prime):
            s_ind = i

    return s_ind

def test_env():
    world = np.zeros(8)
    rewards = []
    game_len = []
    wins = []
    for i in range(10):
        # init the state uniformally over possible states
        s = 71

        # run until game terminates or timeout is reached
        for t in range(50):

            # PRINTING THE WORLD
            world.fill(0)
            world[S[s][1]] = 999
            world[S[s][2]] = 888
            display = world.copy()
            display = np.reshape(display, (2, 4))
            print display

            # choose random action get the set of possible s'
            a_number = np.random.randint(25)

            ball, me, you = S[s]
            a = A[a_number]

            # WE CANNOT GO OVERBOARD!!!
            if (me + a[0] > 7 or me + a[0] < 0):
                a[0] = 0
            if (you + a[1] > 7 or you + a[1] < 0):
                a[1] = 0

            mynewstate = me+a[0]
            yournewstate = me+a[1]

            # GETTING NEW STATE
            if mynewstate == yournewstate:
                first = np.random.randint(2)
                if first == 0: # I'm first
                    if ball == 0: # I have a ball
                        s_prime = [ball, me+a[0], you] # I move, you don't, ball is still mine
                    else:
                        s_prime = [ball-1, me + a[0], you] # i move, you bump into me, I get your ball
                else: # you move first
                    if ball == 1: # you have the ball
                        s_prime = [ball, me, you+a[1]] # you move, you keep the ball, i stay
                    else: # i have the ball
                        s_prime = [ball-1, me, you+a[1]] # you move, I bump into you, you get the ball
            else:
                s_prime = [ball, me+a[0], you+a[1]]
            #
            # s_index = s_prime[0]*56 + s_prime[1]*7 + s_prime[2] - 1
            np_s_prime = np.array(s_prime)

            # print 'index for ', s_prime, ' is ', s_index


            # GET INDEX OF S' STATE
            for i in range(S.shape[0]):
                if np.array_equal(S[i],np_s_prime):
                    s = i
                    # print 'index for ',s_prime, ' is ', i

            #
            # s = s_index

            R_a = R[s, 0]
            R_b = R[s, 1]



            if (R_a != 0 or R_b != 0):
                print "DONE!!!"
                break

# test_env()