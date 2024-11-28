import numpy as np
import math


def policy_improvement_gbike(V, policy, Lamda, lamda, r, t, gam):
    m, n = policy.shape
    nn = np.arange(n)
    P1 = np.exp(-Lamda[0]) * (Lamda[0] ** nn) / np.vectorize(math.factorial)(nn)
    P2 = np.exp(-Lamda[1]) * (Lamda[1] ** nn) / np.vectorize(math.factorial)(nn)
    P3 = np.exp(-lamda[0]) * (lamda[0] ** nn) / np.vectorize(math.factorial)(nn)
    P4 = np.exp(-lamda[1]) * (lamda[1] ** nn) / np.vectorize(math.factorial)(nn)

    policy_stable = True
    old_policy = policy.copy()
    for i in range(m):
        for j in range(n):
            s1 = i - 1
            s2 = j - 1
            v_ = -np.inf
            amin = -min(min(s2, m - 1 - s1), 5)
            amax = min(min(s1, n - 1 - s2), 5)
            for a in range(amin, amax + 1):
                R = -abs(a) * t
                Vs_ = 0
                s1_ = s1 - a
                s2_ = s2 + a
                for n1 in range(13):
                    for n2 in range(15):
                        s1__ = s1_ - min(n1, s1_)
                        s2__ = s2_ - min(n2, s2_)
                        for n3 in range(13):
                            for n4 in range(10):
                                s1___ = s1__ + min(n3, 20 - s1__)
                                s2___ = s2__ + min(n4, 20 - s2__)

                                if 0 <= s1___ < m and 0 <= s2___ < n:
                                    Vs_ += (
                                        P1[n1]
                                        * P2[n2]
                                        * P3[n3]
                                        * P4[n4]
                                        * V[int(s1___), int(s2___)]
                                    )
                                    R += (
                                        P1[n1]
                                        * P2[n2]
                                        * P3[n3]
                                        * P4[n4]
                                        * (min(n1, s1_) + min(n2, s2_))
                                    ) * r
                if R + gam * Vs_ > v_:
                    v_ = R + gam * Vs_
                    policy[i, j] = a
            if np.sum(np.abs(old_policy - policy)) != 0:
                policy_stable = False

    return policy, policy_stable
