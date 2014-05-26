import random
import matplotlib.pyplot as plt

class LCC(object):
    """
    Object of the function

    dXt = (mu)*dt + (sigma)*dWt

    Takes variables:

        S0 - initial asset value (default = 0)
        drift - mu, over a time period of 1 (default = 0)
        diffusion - sigma, over a time period of 1 (default = 0)

    Note that Wt is a Wiener process
    """

    def __init__(self, S0=0.0, drift=0.0, diffusion=0.0):
        self.S0 = S0
        self.drift = drift
        self.diffusion = diffusion


    def plot(self, T=1, n=1000, show_drift=0):
        path = self.sim(T, n)
        xaxis = map(lambda x: x/float(n)*T,range(n+1))
        if show_drift:
            drift_path = map(lambda x: self.S0+(x*self.drift), xaxis)
            plt.plot(xaxis, path, xaxis, drift_path)
            plt.show()
        else:
            plt.plot(xaxis, path)
            plt.show()


    def sim(self, T=1.0, n=1000):
        path = [self.S0]
        h = float(T)/n
        Sinit = self.S0
        for i in xrange(n):
            dWi = random.normalvariate(0, h)
            dSi = (self.drift*h) + (self.diffusion*dWi)
            Si = Sinit + dSi
            path.append(Si)
            Sinit = Si
        return path


class Geometric(object):
    """
    Object of the function

    dXt = (mu)*St*dt + (sigma)*St*dWt

    Takes variables:

        S0 - initial asset value (default = 0)
        drift - mu, over a time period of 1 (default = 0)
        diffusion - sigma, over a time period of 1 (default = 0)
        sqrt_diff - boolean for St or sqrt(St) in diffusion coeff.

    Note that Wt is a Wiener process
    """
    def __init__(self, S0=0.0, drift=0.0, diffusion=0.0, sqrt_diff=0):
        self.S0 = S0
        self.drift = drift
        self.diffusion = diffusion
        self.sqrt_diff = sqrt_diff


    def plot(self, T=1, n=1000, show_drift=0):
        path = self.sim(T, n)
        xaxis = map(lambda x: x/float(n)*T,range(n+1))
        if show_drift:
            drift_path = map(lambda x: self.S0*((1+self.drift)**x), xaxis)
            plt.plot(xaxis, path, xaxis, drift_path)
            plt.show()
        else:
            plt.plot(xaxis, path)
            plt.show()


    def sim(self, T=1.0, n=1000):
        path = [self.S0]
        h = float(T)/n
        Sinit = self.S0
        for i in xrange(n):
            dWi = random.normalvariate(0, h)
            if self.sqrt_diff:
                dSi = (self.drift*Sinit*h) + (self.diffusion*(Sinit**.5)*dWi)
            else:
                dSi = (self.drift*Sinit*h) + (self.diffusion*Sinit*dWi)
            Si = Sinit + dSi
            path.append(Si)
            Sinit = Si
        return path


class MeanRevert(object):
    """
    Object of the function

    dXt = a*((mu)-St)*dt + (sigma)*St*dWt

    Takes variables:

        a - controls mean reversion; higher a means quicker reversion
        S0 - initial asset value (default = 0)
        drift - mu, over a time period of 1 (default = 0)
        diffusion - sigma, over a time period of 1 (default = 0)
        sqrt_diff - boolean for St or sqrt(St) in diffusion coeff.

    Note that Wt is a Wiener process
    """

    def __init__(self, S0=0.0, drift=0.0, diffusion=0.0, sqrt_diff=0, a=.5):
        self.S0 = S0
        self.drift = drift
        self.diffusion = diffusion
        self.sqrt_diff = sqrt_diff
        self.a = a


    def plot(self, T=1, n=1000, show_drift=0):
        path = self.sim(T, n)
        xaxis = map(lambda x: x/float(n)*T,range(n+1))
        if show_drift:
            drift_path = map(lambda x: self.S0*((1+self.drift)**x), xaxis)
            plt.plot(xaxis, path, xaxis, drift_path)
            plt.show()
        else:
            plt.plot(xaxis, path)
            plt.show()


    def sim(self, T=1.0, n=1000):
        path = [self.S0]
        h = float(T)/n
        Si = self.S0
        Smean = Si
        for i in xrange(n):
            dWi = random.normalvariate(0, h)
            Smean = Smean*(1+(h*self.drift))
            if self.sqrt_diff:
                dSi = (self.a*(Smean-Si)*h) \
                        + (self.diffusion*(Si**.5)*dWi)

            else:
                dSi = (self.a*(Smean-Si)*h) \
                        + (self.diffusion*Si*dWi)
            #print Smean, Si, Smean-Si, dWi, Si+dSi
            Si += dSi
            path.append(Si)
        return path



if __name__ == '__main__':
    model = MeanRevert(S0=10, drift=.05, diffusion=.15, sqrt_diff=0, a=2)
    model.plot(T=5,show_drift=1)
